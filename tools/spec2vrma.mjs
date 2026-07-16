// spec2vrma.mjs — モーションspec JSON をコマンドラインから .vrma に変換する
// 使い方: node tools/spec2vrma.mjs input.json output.vrma
import { readFileSync, writeFileSync } from 'node:fs';
import { buildVRMA } from '../src/vrmaBuilder.js';

const [inJson, outVrma] = process.argv.slice(2);
if (!inJson || !outVrma) {
  console.error('usage: node tools/spec2vrma.mjs input.json output.vrma');
  process.exit(1);
}

const spec = JSON.parse(readFileSync(inJson, 'utf-8'));
const glb = buildVRMA(spec);
writeFileSync(outVrma, Buffer.from(glb));
console.log(`OK: ${outVrma} (${glb.byteLength} bytes)`);
