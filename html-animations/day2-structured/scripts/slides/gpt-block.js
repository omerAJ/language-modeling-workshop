const BLOCK32_MAX_STEP = 5;
const BLOCK32_CLASSES = [
  'block32-show-position',
  'block32-show-attn',
  'block32-show-ffn',
  'block32-show-equations',
  'block32-show-summary'
];
const BLOCK32_TAKEAWAYS = [
  'A GPT block updates one shared residual stream with two sublayer writes: attention, then FFN.',
  'The stream entering the block already includes position information.',
  'The first write comes from multi-head attention operating on normalized rows.',
  'The second write comes from the FFN operating on the updated stream.',
  'Once both writes are visible, the compact equations line up with the visual story.',
  'This is the whole block in order: normalize, attend, add, normalize, FFN, add.'
];

function setGptBlockStep(step) {
  const slide = document.getElementById('slide-32');
  const takeaway = document.getElementById('block32-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(BLOCK32_MAX_STEP, step));
  gptBlockState.step = clamped;
  BLOCK32_CLASSES.forEach((className, idx) => {
    slide.classList.toggle(className, clamped >= idx + 1);
  });
  takeaway.innerHTML = BLOCK32_TAKEAWAYS[clamped] || BLOCK32_TAKEAWAYS[0];
  typesetMath(slide);
}

function initGptBlockSlide() {
  gptBlockState.initialized = true;
  setGptBlockStep(gptBlockState.step || 0);
}

function runGptBlockStep() {
  if (!gptBlockState.initialized) initGptBlockSlide();
  if (gptBlockState.step >= BLOCK32_MAX_STEP) return false;
  setGptBlockStep(gptBlockState.step + 1);
  return true;
}

function resetGptBlockSlide() {
  setGptBlockStep(0);
}
