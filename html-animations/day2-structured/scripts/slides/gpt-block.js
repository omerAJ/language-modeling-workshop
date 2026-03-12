const BLOCK30_MAX_STEP = 5;
const BLOCK30_CLASSES = [
  'block30-show-ln',
  'block30-show-attn',
  'block30-show-add',
  'block30-show-ffn',
  'block30-show-eq'
];
const BLOCK30_TAKEAWAYS = [
  'Attention and FFN both write updates into a shared residual stream \u2014 neither one replaces it.',
  'LayerNorm normalizes every row independently before the sublayer sees it.',
  'Multi-head attention gathers context across tokens and produces a small update \\(\\Delta_{\\mathrm{attn}}\\).',
  'The residual add (+) writes that update back into the stream. The original information is preserved.',
  'The FFN transforms each row independently, then a second residual add writes \\(\\Delta_{\\mathrm{ffn}}\\).',
  '\\(R_{\\mathrm{mid}} = R^{(\\ell)} + \\mathrm{MHA}(\\mathrm{LN}(R^{(\\ell)}))\\) &nbsp; \\(R^{(\\ell+1)} = R_{\\mathrm{mid}} + \\mathrm{FFN}(\\mathrm{LN}(R_{\\mathrm{mid}}))\\)'
];

function setGptBlockStep(step) {
  const slide = document.getElementById('slide-30');
  const takeaway = document.getElementById('block30-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(BLOCK30_MAX_STEP, step));
  gptBlockState.step = clamped;
  BLOCK30_CLASSES.forEach((className, idx) => {
    slide.classList.toggle(className, clamped >= idx + 1);
  });
  takeaway.innerHTML = BLOCK30_TAKEAWAYS[clamped] || BLOCK30_TAKEAWAYS[0];
}

function initGptBlockSlide() {
  setGptBlockStep(gptBlockState.step || 0);
  typesetMath(document.getElementById('slide-30'));
}

function runGptBlockStep() {
  if (gptBlockState.step >= BLOCK30_MAX_STEP) return false;
  setGptBlockStep(gptBlockState.step + 1);
  return true;
}

function resetGptBlockSlide() {
  setGptBlockStep(0);
}
