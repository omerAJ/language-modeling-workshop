const RESID30_MAX_STEP = 5;
const RESID30_CLASSES = [
  'resid30-show-attn',
  'resid30-show-attn-merge',
  'resid30-show-ffn',
  'resid30-show-ffn-merge',
  'resid30-show-equations'
];
const RESID30_TAKEAWAYS = [
  'The residual stream persists; sublayers compute updates and write them back.',
  'Start with the attention branch: it builds an update instead of replacing the stream.',
  'After the attention write, the stream continues forward as \\(R_{\\mathrm{mid}}\\).',
  'The FFN does the same thing: compute another update to write into the same stream.',
  'The output of the block is the residual stream after both writes.',
  'These equations are the compact form of the write-back story you just watched.'
];

function setResidualStreamStep(step) {
  const slide = document.getElementById('slide-30');
  const takeaway = document.getElementById('resid30-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(RESID30_MAX_STEP, step));
  residualStreamState.step = clamped;
  RESID30_CLASSES.forEach((className, idx) => {
    slide.classList.toggle(className, clamped >= idx + 1);
  });
  takeaway.innerHTML = RESID30_TAKEAWAYS[clamped] || RESID30_TAKEAWAYS[0];
  typesetMath(slide);
}

function initResidualStreamSlide() {
  residualStreamState.initialized = true;
  setResidualStreamStep(residualStreamState.step || 0);
}

function runResidualStreamStep() {
  if (!residualStreamState.initialized) initResidualStreamSlide();
  if (residualStreamState.step >= RESID30_MAX_STEP) return false;
  setResidualStreamStep(residualStreamState.step + 1);
  return true;
}

function resetResidualStreamSlide() {
  setResidualStreamStep(0);
}
