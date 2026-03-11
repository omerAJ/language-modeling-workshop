const POS27_MAX_STEP = 4;
const POS27_CLASSES = [
  'pos27-show-intuition',
  'pos27-show-linearity',
  'pos27-show-caveat',
  'pos27-show-footer'
];
const POS27_TAKEAWAYS = [
  'Adding position raises a fair question: if we sum token and slot information, how can the model still use them separately?',
  'The sum is a composed state: one reusable content vector combines with reusable slot vectors to produce position-aware rows.',
  'Because projection is linear, (x_i + p_i)W = x_iW + p_iW. The next attention layer can respond to content and slot contributions separately.',
  'Exact collisions are possible in principle, but exact inversion is not the goal. The model only needs the composed rows to remain useful for downstream prediction.',
  'Additive position is the simple teaching model. Many modern decoder LMs instead inject position through rotary methods.'
];

function setPositionSuperpositionStep(step) {
  const slide = document.getElementById('slide-27');
  const takeaway = document.getElementById('pos27-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(POS27_MAX_STEP, step));
  positionSuperpositionState.step = clamped;
  POS27_CLASSES.forEach((className, idx) => {
    slide.classList.toggle(className, clamped >= idx + 1);
  });
  takeaway.innerHTML = POS27_TAKEAWAYS[clamped] || POS27_TAKEAWAYS[0];
  typesetMath(slide);
}

function initPositionSuperpositionSlide() {
  positionSuperpositionState.initialized = true;
  setPositionSuperpositionStep(positionSuperpositionState.step || 0);
}

function runPositionSuperpositionStep() {
  if (!positionSuperpositionState.initialized) initPositionSuperpositionSlide();
  if (positionSuperpositionState.step >= POS27_MAX_STEP) return false;
  setPositionSuperpositionStep(positionSuperpositionState.step + 1);
  return true;
}

function resetPositionSuperpositionSlide() {
  setPositionSuperpositionStep(0);
}
