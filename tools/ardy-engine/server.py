# server.py — ARDY ローカルエンジンサーバー (text-to-vrma 用)
#
# NVIDIA ARDY (Autoregressive Diffusion for Interactive Motion) を常駐させ、
# アプリからHTTPで叩けるようにする。日本語プロンプトはローカル翻訳 (FuguMT) で自動英訳。
#
# 起動例:
#   python server.py --merged-base C:\path\to\llm2vec-base-merged
#   (テキストエンコーダをCPUで動かす場合は環境変数 TEXT_ENCODER_DEVICE=cpu)
#
# API:
#   GET  /health   → {"status":"ok","model":...,"translator":...}
#   POST /generate body: {"text":"...", "duration":5.0, "seed":0(省略可)}
#                  → モーションspec JSON (buildVRMA互換)。日本語入力時は
#                    spec.name が英訳、spec.originalText に原文が入る
import argparse
import importlib
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

JP_RE = re.compile(r"[぀-ヿ一-鿿]")
TRANSLATOR_MODEL = "staka/fugumt-ja-en"


def parse_args():
    ap = argparse.ArgumentParser(description="ARDY local engine server for text-to-vrma")
    ap.add_argument("--port", type=int, default=int(os.environ.get("ARDY_PORT", 2337)))
    ap.add_argument("--model", default="ARDY-Core-RP-20FPS-Horizon40")
    ap.add_argument(
        "--merged-base",
        default=os.environ.get("ARDY_MERGED_BASE", ""),
        help="mntpマージ済みLlama-3-8Bのローカルパス (未指定なら公式gatedリポジトリ経由)",
    )
    ap.add_argument("--no-translate", action="store_true", help="日本語自動英訳を無効化")
    ap.add_argument(
        "--arm-spread", type=float, default=6.0,
        help="上腕を外側に開くオフセット (度)。アニメ体型への腕のめり込み対策 (既定: 6)",
    )
    ap.add_argument(
        "--history", type=int, default=40,
        help="自己回帰の履歴フレーム数。長いと文脈が滑らかだが、動きの慣性が勝って"
             "テキスト追従が落ちる (NVIDIAデモの既定は最小=4)。既定: 40",
    )
    return ap.parse_args()


ARGS = parse_args()

# --- モデルロード (プロセスで1回) ---
print("loading ARDY model... (first time: 1-2 min)", flush=True)
import torch  # noqa: E402

if ARGS.merged_base:
    lm = importlib.import_module("ardy.model.load_model")
    lm.TEXT_ENCODER_PRESETS["llm2vec"]["kwargs"]["base_model_name_or_path"] = ARGS.merged_base

from ardy.constraints import Root2DConstraintSet  # noqa: E402
from ardy.model import load_model  # noqa: E402
from ardy.postprocess import post_process_motion  # noqa: E402
from ardy.tools import seed_everything, to_numpy  # noqa: E402

from retarget import spec_from_arrays  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model(ARGS.model, device=DEVICE)
FPS = model.motion_rep.fps
PATCH = model.num_frames_per_token
NUM_BASE_STEPS = int(model.diffusion.num_base_steps)
# 履歴上限 (訓練10秒窓 - ホライズン)。実際の履歴長は --history (既定40) で制御
_max_window = (int(10 * FPS) // PATCH) * PATCH
_history_cap = ((_max_window - model.gen_horizon_len) // PATCH) * PATCH
HISTORY = max(PATCH, min(_history_cap, (ARGS.history // PATCH) * PATCH))
GEN_LOCK = threading.Lock()
print(f"ready: {ARGS.model} on {DEVICE} ({FPS} fps)", flush=True)

# --- 日本語→英語翻訳 (FuguMT、バックグラウンドでロード) ---
_translator = None
_translator_state = "disabled" if ARGS.no_translate else "loading"


def _load_translator():
    # transformers 5.x では translation パイプラインが廃止されたため、MarianMTを直接使う
    global _translator, _translator_state
    try:
        from transformers import MarianMTModel, MarianTokenizer

        tok = MarianTokenizer.from_pretrained(TRANSLATOR_MODEL)
        mt = MarianMTModel.from_pretrained(TRANSLATOR_MODEL)
        mt.eval()

        def translate(text: str) -> str:
            batch = tok([text], return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                out = mt.generate(**batch, max_length=256, num_beams=4)
            return tok.decode(out[0], skip_special_tokens=True)

        _translator = translate
        _translator_state = "ready"
        print(f"translator ready: {TRANSLATOR_MODEL}", flush=True)
    except Exception as e:  # noqa: BLE001
        _translator_state = f"error: {e}"
        print(f"translator failed to load: {e}", flush=True)


if not ARGS.no_translate:
    threading.Thread(target=_load_translator, daemon=True).start()


def maybe_translate(text: str):
    """日本語を含むテキストを英訳して (english, original|None) を返す。"""
    if not JP_RE.search(text):
        return text, None
    if _translator is None:
        if _translator_state == "loading":
            # 初回リクエストがロードと競合した場合は最大60秒待つ
            for _ in range(60):
                if _translator is not None or _translator_state.startswith("error"):
                    break
                time.sleep(1)
        if _translator is None:
            raise RuntimeError(f"日本語翻訳を利用できません ({_translator_state})")
    english = _translator(text).strip()
    print(f"translated: '{text}' -> '{english}'", flush=True)
    return english, text


MAX_DURATION = 60.0

# --- 進捗トラッキング ---
# stage: translate (翻訳・エンコード) → generate (チャンク生成、fractionが本当の進捗) → finalize (変換)
PROGRESS = {"active": False, "stage": "", "started": 0.0, "fraction": 0.0, "text": ""}


def estimate_duration(english: str, original: str | None) -> float:
    """プロンプト内容からモーション長 (秒) を推定する。"""
    t = english.lower()
    base = 5.0
    # 連続動作 (「〜してから〜」等) は接続表現の数だけ延長
    seq = len(re.findall(r"\b(then|after that|afterwards|next|followed by)\b|,| and ", t))
    base += 3.0 * min(seq, 6)
    # 移動・リズム系はサイクルを見せるため長めに
    if re.search(r"danc|run|jog|walk|march|stroll|wander|sneak|stagger", t):
        base += 4.0
    if re.search(r"in a circle|around|back and forth|repeatedly|keep", t):
        base += 3.0
    if re.search(r"slowly|slow|gradually", t):
        base += 2.0
    # 単発ジェスチャは短く
    if seq == 0 and re.search(r"^\s*(a person )?(wave|bow|nod|clap|salute|point)s?\b", t):
        base = 3.5
    return max(3.0, min(MAX_DURATION, base))


VRM_SCALE = 0.9 / 0.896  # retarget.py と同じ (アプリ座標 = Core座標 × scale)


def build_waypoint_conditions(waypoints, num_frames):
    """クリック経由地 [{x,z}...] をARDYのルート2D制約に変換する。

    時刻はパスの累積距離に比例して 1秒〜終端-0.5秒 に配分する。
    Returns (observed_motion, motion_mask) — フル長 [1, T, F]。
    """
    pts = [(float(w["x"]) / VRM_SCALE, float(w["z"]) / VRM_SCALE) for w in waypoints]
    dists = [0.0]
    prev = (0.0, 0.0)  # キャラは原点から出発
    for p in pts:
        dists.append(dists[-1] + ((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2) ** 0.5)
        prev = p
    total = max(dists[-1], 1e-6)
    # 開始1秒後 〜 終端2秒前に距離比例で配分
    t0 = FPS
    t1 = max(t0 + PATCH, num_frames - 2 * FPS)
    frame_indices = [
        min(num_frames - 1, int(t0 + (t1 - t0) * (dists[i + 1] / total)))
        for i in range(len(pts))
    ]
    # 最終経由地に「留まる」制約: 終端までルートを同じ位置にピン留めして、
    # モデル自身に減速→停止させる (これがないとテキスト通り歩き続けて
    # 歩行姿勢のままクリップが終わる)
    for f in (num_frames - FPS, num_frames - PATCH):
        if f > frame_indices[-1]:
            frame_indices.append(f)
            pts.append(pts[-1])
    constraint = Root2DConstraintSet(
        model.skeleton,
        frame_indices=torch.tensor(frame_indices),
        root_2d=torch.tensor(pts, dtype=torch.float32, device=DEVICE),
    )
    lengths = torch.tensor([num_frames], device=DEVICE)
    observed, mask = model.motion_rep.create_conditions_from_constraints_batched(
        [constraint], lengths, to_normalize=True, device=DEVICE,
    )
    return observed, mask


def waypoint_duration(waypoints) -> float:
    """経由地の合計距離から歩行前提の所要時間を見積もる (開始1秒+移動+停止余白2秒)。"""
    prev, dist = (0.0, 0.0), 0.0
    for w in waypoints:
        p = (float(w["x"]), float(w["z"]))
        dist += ((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2) ** 0.5
        prev = p
    return max(4.0, min(MAX_DURATION, dist / 1.0 + 3.0))


def _generate_motion_streaming(segments: list, steps: int, observed=None, mask=None, on_chunk=None):
    """デモと同じチャンク逐次生成。セグメントごとにプロンプトを切り替える。

    バッチAPI (model.__call__) は各チャンクで残り全フレームを処理するため
    長尺で二次関数的に遅くなる。autoregressive_step で「履歴+1ホライズン」だけを
    処理すればコストは長さに比例し、チャンクごとに本当の進捗も取れる。
    プロンプトはチャンク開始フレームが属するセグメントのものを使う (ARDYの
    ストリーミングテキスト条件付けと同じ仕組み)。

    Args:
        segments: [(english_text, num_frames), ...]
    """
    horizon = model.gen_horizon_len
    feats = [model._encode_text([en]) for en, _ in segments]
    bounds = []  # セグメント開始フレーム
    acc = 0
    for _, nf in segments:
        bounds.append(acc)
        acc += nf
    num_frames = acc
    motion = None  # 正規化済みハイブリッドモーション [1, T, F]
    while motion is None or motion.shape[1] < num_frames:
        pos = 0 if motion is None else motion.shape[1]
        seg_idx = max(i for i, b in enumerate(bounds) if b <= pos)
        text_feat, text_pad_mask = feats[seg_idx]
        PROGRESS["text"] = segments[seg_idx][0]
        if motion is None:
            hist, hist_len = None, 0
            init_transl = torch.zeros(1, model.motion_rep.nfeats_dict["root_pos"], device=DEVICE)
            init_heading = torch.zeros(1, device=DEVICE)
        else:
            # セグメント切替直後は履歴を短くして新プロンプトへ素早く遷移させる
            # (長い履歴は直前の動きの慣性が勝ち、テキスト切替が効かない)。
            # 単一セグメントでは従来通り HISTORY まで履歴を使う
            frames_since_seg = pos - bounds[seg_idx]
            max_hist = min(HISTORY, frames_since_seg + PATCH * 2)
            hist_len = min(motion.shape[1], max_hist) // PATCH * PATCH
            hist_len = max(hist_len, PATCH)
            hist = motion[:, -hist_len:]
            init_transl, init_heading = None, None
        # 未来領域: 制約 (ウェイポイント等) を先読みさせるための空きトークン。
        # 訓練10秒窓に収める
        remaining = max(0, num_frames - (pos + horizon))
        future_len = min(remaining, _max_window - hist_len - horizon)
        future_len = max(0, future_len // PATCH * PATCH)
        window_frames = hist_len + horizon + future_len
        # ウェイポイント制約: 窓に対応する区間を切り出し、履歴部分は無効化
        # (デモ GenerationMixin._generate_step と同じ扱い)
        win_mask = win_observed = None
        if mask is not None:
            start = pos - hist_len
            win_mask = mask[:, start:start + window_frames].clone()
            win_observed = observed[:, start:start + window_frames].clone()
            if win_mask.shape[1] < window_frames:  # 終端で足りない分はゼロ埋め
                pad = window_frames - win_mask.shape[1]
                win_mask = torch.nn.functional.pad(win_mask, (0, 0, 0, pad))
                win_observed = torch.nn.functional.pad(win_observed, (0, 0, 0, pad))
            win_mask[:, :hist_len] = 0.0
            win_observed[:, :hist_len] = 0.0
        samples = model.autoregressive_step(
            num_frames=window_frames,
            num_denoising_steps=steps,
            motion_mask=win_mask,
            observed_motion=win_observed,
            cfg_weight=(2.0, 2.0),
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            init_history_sequence=hist,
            init_global_translation=init_transl,
            init_first_heading_angle=init_heading,
        )
        new = samples[:, hist_len:]
        # チャンクをセグメント境界に整列: 境界を越えた分は捨てて、
        # 次のチャンクが新セグメントのプロンプトで境界ちょうどから始まるようにする
        next_bound = min((b for b in bounds if b > pos), default=num_frames)
        keep = max(PATCH, (next_bound - pos) // PATCH * PATCH) if next_bound - pos < new.shape[1] else new.shape[1]
        new = new[:, :keep]
        motion = new if motion is None else torch.cat([motion, new], dim=1)
        PROGRESS["fraction"] = min(0.99, motion.shape[1] / num_frames)
        if on_chunk is not None:
            end = min(motion.shape[1], num_frames)
            if end > pos:
                on_chunk(motion[:, pos:end], pos, num_frames)
    return motion[:, :num_frames]


def _resolve_segments(text, duration, segments_req):
    """リクエストを [(english, num_frames), ...] と表示名に正規化する。"""
    if segments_req:
        segments, originals = [], []
        for seg in segments_req[:12]:
            seg_text = str(seg.get("text", "")).strip()
            if not seg_text:
                continue
            english, original = maybe_translate(seg_text)
            seg_dur = seg.get("duration") or estimate_duration(english, original)
            nf = max(PATCH, int(float(seg_dur) * FPS) // PATCH * PATCH)
            segments.append((english, nf))
            originals.append(original or english)
        if not segments:
            raise ValueError("segments に有効なテキストがありません")
        # 合計を最長60秒に収める (超過分は末尾から比例縮小)
        total = sum(nf for _, nf in segments)
        max_frames = int(MAX_DURATION * FPS)
        if total > max_frames:
            ratio = max_frames / total
            segments = [(en, max(PATCH, int(nf * ratio) // PATCH * PATCH)) for en, nf in segments]
        name = " / ".join(en for en, _ in segments)
        return segments, name, None
    english, original = maybe_translate(text.strip())
    if duration is None:
        duration = estimate_duration(english, original)
        print(f"auto duration: {duration:.1f}s for '{english[:50]}'", flush=True)
    duration = max(0.5, min(MAX_DURATION, float(duration)))
    num_frames = max(PATCH, int(duration * FPS) // PATCH * PATCH)
    return [(english, num_frames)], english, original


def generate_spec(
    text: str, duration=None, seed=None, steps=None, arm_spread=None, segments_req=None,
    postprocess=True, waypoints=None, on_chunk=None,
) -> dict:
    arm_spread = ARGS.arm_spread if arm_spread is None else max(0.0, min(20.0, float(arm_spread)))
    steps = int(steps) if steps else NUM_BASE_STEPS
    steps = max(1, min(NUM_BASE_STEPS, steps))
    with GEN_LOCK, torch.no_grad():
        PROGRESS.update(active=True, stage="translate", started=time.time(), fraction=0.0, text=text)
        try:
            # ウェイポイントがあれば、経路距離から所要時間の下限を決める
            if waypoints and duration is None and not segments_req:
                duration = waypoint_duration(waypoints)
                print(f"waypoint duration: {duration:.1f}s ({len(waypoints)} points)", flush=True)
            segments, english, original = _resolve_segments(text, duration, segments_req)
            num_frames = sum(nf for _, nf in segments)
            # セグメント計画 (GPT) が経由地の経路に対して短すぎる場合は比例拡大する
            # (短いままだと経由地間を無理な速度で移動しようとして破綻する)
            if waypoints:
                needed = int(waypoint_duration(waypoints) * FPS) // PATCH * PATCH
                if num_frames < needed:
                    ratio = needed / num_frames
                    segments = [
                        (en, max(PATCH, int(nf * ratio) // PATCH * PATCH)) for en, nf in segments
                    ]
                    num_frames = sum(nf for _, nf in segments)
                    print(
                        f"segments scaled x{ratio:.1f} for waypoints -> {num_frames / FPS:.1f}s",
                        flush=True,
                    )
            if len(segments) > 1:
                plan = ", ".join(f"'{en[:30]}'x{nf}f" for en, nf in segments)
                print(f"segments: {plan}", flush=True)
            observed = cmask = None
            if waypoints:
                observed, cmask = build_waypoint_conditions(waypoints, num_frames)
            if seed is not None:
                seed_everything(int(seed))
            PROGRESS.update(stage="generate", started=time.time(), text=original or english)
            t_model = time.time()
            motion = _generate_motion_streaming(segments, steps, observed, cmask, on_chunk)
            t_inverse = time.time()
            PROGRESS["stage"] = "finalize"
            output = model.motion_rep.inverse(motion, is_normalized=True)
            if postprocess:
                corrected = post_process_motion(
                    output["local_rot_mats"],
                    output["root_positions"],
                    output["foot_contacts"],
                    model.skeleton,
                )
                output.update(corrected)
            t_done = time.time()
            print(
                f"timing: generate={t_inverse-t_model:.1f}s finalize={t_done-t_inverse:.1f}s "
                f"({num_frames}f, {steps} steps)",
                flush=True,
            )
            output = to_numpy(output)
            spec = spec_from_arrays(
                output["global_rot_mats"], output["root_positions"], FPS, english,
                arm_spread_deg=arm_spread,
            )
            if original is not None:
                spec["originalText"] = original
            return spec
        finally:
            PROGRESS["active"] = False


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {
                "status": "ok",
                "model": ARGS.model,
                "device": DEVICE,
                "fps": FPS,
                "translator": _translator_state,
            })
        elif self.path == "/progress":
            if PROGRESS["active"]:
                elapsed = time.time() - PROGRESS["started"]
                frac = PROGRESS["fraction"]
                # 残り時間は実進捗ベースで推定 (fraction=生成済みフレーム/総フレーム)
                remaining = (elapsed / frac - elapsed) if frac > 0.02 else None
                self._send(200, {
                    "active": True,
                    "stage": PROGRESS["stage"],
                    "elapsed": round(elapsed, 1),
                    "remaining": None if remaining is None else round(remaining, 1),
                    "fraction": round(frac, 3),
                    "text": PROGRESS["text"],
                })
            else:
                self._send(200, {"active": False})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/generate-stream":
            self._handle_generate_stream()
            return
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            text = str(req.get("text", "")).strip()
            segments_req = req.get("segments")
            duration = req.get("duration")  # 省略時はプロンプトから自動推定
            if not text and not segments_req:
                self._send(400, {"error": "text or segments is required"})
                return
            t0 = time.time()
            spec = generate_spec(
                text, duration, req.get("seed"), req.get("steps"), req.get("armSpread"),
                segments_req, req.get("postprocess", True), req.get("waypoints"),
            )
            print(f"generated '{spec['name'][:50]}' ({spec['duration']}s) in {time.time()-t0:.1f}s", flush=True)
            self._send(200, spec)
        except Exception as e:  # noqa: BLE001 — クライアントにエラーを返して継続
            import traceback

            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def _handle_generate_stream(self):
        """NDJSONストリーミング生成: チャンクができるたびにspec断片を送る。

        行の種類: {"type":"meta",...} → {"type":"chunk",tracks,hips}× → {"type":"final","spec":...}
        エラー時は {"type":"error","error":...}
        """
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": str(e)})
            return
        text = str(req.get("text", "")).strip()
        segments_req = req.get("segments")
        if not text and not segments_req:
            self._send(400, {"error": "text or segments is required"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(obj):
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        arm = req.get("armSpread")
        arm = ARGS.arm_spread if arm is None else max(0.0, min(20.0, float(arm)))
        state = {"meta_sent": False}

        def on_chunk(chunk, pos, total):
            if not state["meta_sent"]:
                emit({"type": "meta", "fps": FPS, "numFrames": total, "duration": round(total / FPS, 3)})
                state["meta_sent"] = True
            out = to_numpy(model.motion_rep.inverse(chunk, is_normalized=True))
            frag = spec_from_arrays(
                out["global_rot_mats"], out["root_positions"], FPS, "chunk",
                arm_spread_deg=arm, t_offset=pos / FPS,
            )
            emit({"type": "chunk", "tracks": frag["tracks"], "hips": frag["hips"]})

        try:
            t0 = time.time()
            spec = generate_spec(
                text, req.get("duration"), req.get("seed"), req.get("steps"), arm,
                segments_req, req.get("postprocess", True), req.get("waypoints"),
                on_chunk=on_chunk,
            )
            print(f"stream-generated '{spec['name'][:50]}' in {time.time()-t0:.1f}s", flush=True)
            emit({"type": "final", "spec": spec})
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            try:
                emit({"type": "error", "error": str(e)})
            except Exception:  # noqa: BLE001 — クライアント切断時
                pass

    def log_message(self, *args):
        pass  # アクセスログは抑制 (生成ログだけ出す)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"ARDY engine server listening on http://127.0.0.1:{ARGS.port}", flush=True)
    server.serve_forever()
