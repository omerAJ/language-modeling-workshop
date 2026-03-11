const LN31_MAX_STEP = 4;
const LN31_CLASSES = [
  'ln31-show-box',
  'ln31-show-row-a',
  'ln31-show-row-b',
  'ln31-show-targets'
];
const LN31_TAKEAWAYS = [
  'LayerNorm is a per-row preparation step, not another token-mixing step.',
  'First identify the problem: wildly different row scales would hit the next sublayer differently.',
  'LayerNorm prepares one row at a time before the next module sees it.',
  'A second row is normalized independently, not by mixing it with the first.',
  'Both attention and the FFN receive row-wise normalized inputs in this pre-norm block.'
];

function setLayerNormStoryStep(step) {
  const slide = document.getElementById('slide-31');
  const takeaway = document.getElementById('ln31-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(LN31_MAX_STEP, step));
  layerNormStoryState.step = clamped;
  LN31_CLASSES.forEach((className, idx) => {
    slide.classList.toggle(className, clamped >= idx + 1);
  });
  takeaway.textContent = LN31_TAKEAWAYS[clamped] || LN31_TAKEAWAYS[0];
}

function initLayerNormStorySlide() {
  layerNormStoryState.initialized = true;
  setLayerNormStoryStep(layerNormStoryState.step || 0);
}

function runLayerNormStoryStep() {
  if (!layerNormStoryState.initialized) initLayerNormStorySlide();
  if (layerNormStoryState.step >= LN31_MAX_STEP) return false;
  setLayerNormStoryStep(layerNormStoryState.step + 1);
  return true;
}

function resetLayerNormStorySlide() {
  setLayerNormStoryStep(0);
}
