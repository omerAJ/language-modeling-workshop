function devLog() {
  if (!DEV) return;
  console.log.apply(console, ['[DEV]'].concat(Array.from(arguments)));
}

function devAssert(condition, message, context) {
  if (!DEV || condition) return;
  if (typeof context === 'undefined') {
    console.warn('[DEV][ASSERT]', message);
    return;
  }
  console.warn('[DEV][ASSERT]', message, context);
}

function captureSnapshot(label) {
  if (!DEV) return;
  const activeSlide = state.nav.slides[state.nav.current];
  const snapshot = {
    label,
    currentIndex: state.nav.current,
    currentSlideId: activeSlide ? activeSlide.id : null,
    projectionLens: projectionState.activeLens,
    attentionIntroStep: attentionIntroState.step,
    attentionP1Step: attentionP1State.step,
    attentionQkvStep: attentionQkvState.step,
    attentionWeightsStep: attentionWeightsState.step,
    attentionStep4Step: attentionStep4State.step,
    attentionMatrixStep: attentionMatrixState.step,
    attentionMultiHeadStep: attentionMultiHeadState.step,
    attentionPositionStep: attentionPositionState.step,
    orderProblemStep: orderProblemState.step,
    positionSignalStep: positionSignalState.step,
    gptBlockStep: gptBlockState.step,
    outputHeadStep: outputHeadState.step,
    generationStep: generationState.step
  };
  state.dev.snapshots.push(snapshot);
  devLog('snapshot', snapshot);
}

function runDevStartupChecks() {
  if (!DEV) return;
  const expectedOrder = [
    'slide-0', 'slide-1', 'slide-2', 'slide-3', 'slide-4',
    'slide-5', 'slide-6', 'slide-7', 'slide-8', 'slide-9',
    'slide-10', 'slide-11', 'slide-12', 'slide-13', 'slide-14',
    'slide-16', 'slide-17', 'slide-18', 'slide-19', 'slide-20',
    'slide-21', 'slide-22', 'slide-23', 'slide-24', 'slide-25',
    'slide-26', 'slide-27', 'slide-28',
    'slide-31', 'slide-32', 'slide-33'
  ];
  const foundOrder = state.nav.slides.map((el) => el.id);
  devAssert(state.nav.total === 31, 'Slide count mismatch', { expected: 31, actual: state.nav.total });
  devAssert(
    expectedOrder.join('|') === foundOrder.join('|'),
    'Slide order mismatch',
    { expected: expectedOrder, actual: foundOrder }
  );
  devAssert(Boolean(state.ui.btnPrev), 'Missing #btnPrev');
  devAssert(Boolean(state.ui.btnNext), 'Missing #btnNext');
  devAssert(Boolean(state.ui.btnSkip), 'Missing #btnSkip');
  devAssert(Boolean(state.ui.progressFill), 'Missing #progressFill');
  devAssert(Boolean(state.ui.slideCounter), 'Missing #slideCounter');
  devLog('listener counts', state.dev.listenerCounts);
}
