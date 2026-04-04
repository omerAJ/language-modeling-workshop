const BLOCK28_MAX_STEP = 5;
const BLOCK28_CLASSES = [
  'block28-show-ln',
  'block28-show-attn',
  'block28-show-add',
  'block28-show-ffn',
  'block28-show-eq'
];
const BLOCK28_TAKEAWAYS = [
  'One block = LayerNorm, attention, residual add, LayerNorm, FFN, residual add.',
  'LayerNorm normalizes each row before a sublayer reads it.',
  'Attention mixes across tokens and writes a context update.',
  'Residual add writes that update back into the same stream.',
  'The FFN updates each row independently, then writes its own residual update.',
  'A full model stacks this same Transformer block layer after layer.'
];

function setGptBlockStep(step) {
  const slide = document.getElementById('slide-28');
  const takeaway = document.getElementById('block28-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(BLOCK28_MAX_STEP, step));
  gptBlockState.step = clamped;
  BLOCK28_CLASSES.forEach((className, idx) => {
    slide.classList.toggle(className, clamped >= idx + 1);
  });
  takeaway.innerHTML = BLOCK28_TAKEAWAYS[clamped] || BLOCK28_TAKEAWAYS[0];
}

function initGptBlockSlide() {
  setGptBlockStep(gptBlockState.step || 0);
  typesetMath(document.getElementById('slide-28'));
}

function runGptBlockStep() {
  if (gptBlockState.step >= BLOCK28_MAX_STEP) return false;
  setGptBlockStep(gptBlockState.step + 1);
  return true;
}

function resetGptBlockSlide() {
  setGptBlockStep(0);
}
