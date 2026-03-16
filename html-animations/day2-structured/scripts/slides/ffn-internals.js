const FFN29_MAX_STEP = 5;
const FFN29_CLASSES = [
  'ffn29-show-input',
  'ffn29-show-expand',
  'ffn29-show-gelu',
  'ffn29-show-project',
  'ffn29-show-math'
];
const FFN29_TAKEAWAYS = [
  'The FFN starts from one contextualized token row.',
  'The input row \\(h_i\\) enters the FFN.',
  'The first linear layer expands from \\(d_{\\mathrm{model}}\\) to \\(d_{\\mathrm{ff}}\\).',
  'The middle-layer neurons apply GELU, shown with a simplified activation marker.',
  'The second linear layer projects back from \\(d_{\\mathrm{ff}}\\) to \\(d_{\\mathrm{model}}\\).',
  'The full FFN is a local computation: \\(\\mathrm{FFN}(h_i) = W_2\\,\\mathrm{GELU}(W_1 h_i + b_1) + b_2\\), reused independently at every position.'
];

function setFfnInternalsStep(step) {
  const slide = document.getElementById('slide-29');
  const takeaway = document.getElementById('ffn29-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(FFN29_MAX_STEP, step));
  ffnInternalsState.step = clamped;
  FFN29_CLASSES.forEach((className, idx) => {
    slide.classList.toggle(className, clamped >= idx + 1);
  });
  takeaway.innerHTML = FFN29_TAKEAWAYS[clamped] || FFN29_TAKEAWAYS[0];
  typesetMath(slide);
}

function initFfnInternalsSlide() {
  ffnInternalsState.initialized = true;
  setFfnInternalsStep(ffnInternalsState.step || 0);
}

function runFfnInternalsStep() {
  if (!ffnInternalsState.initialized) initFfnInternalsSlide();
  if (ffnInternalsState.step >= FFN29_MAX_STEP) return false;
  setFfnInternalsStep(ffnInternalsState.step + 1);
  return true;
}

function resetFfnInternalsSlide() {
  setFfnInternalsStep(0);
}
