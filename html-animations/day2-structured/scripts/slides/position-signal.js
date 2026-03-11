const POS26_MAX_STEP = 5;
const POS26_CLASSES = [
  'pos26-show-setup',
  'pos26-show-left',
  'pos26-show-right',
  'pos26-show-compare',
  'pos26-show-formula'
];
const POS26_TAKEAWAYS = [
  'Position is the missing ingredient: without it, the same word looks the same no matter where it appears.',
  'We give every word a token vector and every slot its own position vector.',
  'In the original order, cat sits in slot 3, so its input row becomes x_cat + p_3 = [4,2].',
  'After permutation, cat moves to slot 1, so its row changes to x_cat + p_1 = [4,1].',
  'Now the same word is not the same row anymore: cat in slot 3 and cat in slot 1 are different inputs to attention.',
  'These position-aware rows are what attention projects into Q, K, and V next. The next question is: after addition, why does the content signal not get lost?'
];

function setPositionSignalStep(step) {
  const slide = document.getElementById('slide-26');
  const takeaway = document.getElementById('pos26-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(POS26_MAX_STEP, step));
  positionSignalState.step = clamped;
  POS26_CLASSES.forEach((className, idx) => {
    slide.classList.toggle(className, clamped >= idx + 1);
  });
  takeaway.innerHTML = POS26_TAKEAWAYS[clamped] || POS26_TAKEAWAYS[0];
  typesetMath(slide);
}

function initPositionSignalSlide() {
  positionSignalState.initialized = true;
  setPositionSignalStep(positionSignalState.step || 0);
}

function runPositionSignalStep() {
  if (!positionSignalState.initialized) initPositionSignalSlide();
  if (positionSignalState.step >= POS26_MAX_STEP) return false;
  setPositionSignalStep(positionSignalState.step + 1);
  return true;
}

function resetPositionSignalSlide() {
  setPositionSignalStep(0);
}
