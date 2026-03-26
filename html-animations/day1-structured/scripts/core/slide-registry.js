function createSlideDescriptor(id, overrides) {
  const base = {
    id,
    build: () => {},
    init: () => {},
    step: () => false,
    reset: () => {}
  };
  return Object.assign(base, overrides || {});
}

function defineSlides() {
  const descriptors = [];
  const interactive = {};

  Object.assign(interactive, {
    'slide-3': createSlideDescriptor('slide-3', {
      init: () => resetIngredients(),
      step: () => runIngredientsStep(),
      reset: () => resetIngredients()
    }),
    'slide-5': createSlideDescriptor('slide-5', {
      init: () => resetInferenceDemo(),
      step: () => runInferenceSlideStep(),
      reset: () => resetInferenceDemo()
    }),
    'slide-6': createSlideDescriptor('slide-6', {
      init: () => resetTrainLoop(),
      step: () => runTrainingLoopStep(),
      reset: () => resetTrainLoop()
    }),
    'slide-10': createSlideDescriptor('slide-10', {
      init: () => initPredictedChartSlide()
    }),
    'slide-11': createSlideDescriptor('slide-11', {
      init: () => initTargetComparisonSlide()
    }),
    'slide-12': createSlideDescriptor('slide-12', {
      init: () => initLossComputationSlide()
    }),
    'slide-14': createSlideDescriptor('slide-14', {
      init: () => initAfterUpdateSlide()
    }),
    'slide-16': createSlideDescriptor('slide-16', {
      step: () => runNtpSlideStep()
    }),
    'slide-22': createSlideDescriptor('slide-22', {
      init: () => resetReasoningPressureSlide(),
      step: () => runReasoningPressureStep(),
      reset: () => resetReasoningPressureSlide()
    }),
    'slide-23': createSlideDescriptor('slide-23', {
      init: () => resetDramaSlide(),
      step: () => runDramaSlideStep(),
      reset: () => resetDramaSlide()
    }),
    'slide-31': createSlideDescriptor('slide-31', {
      init: () => resetCompletionActivity(),
      step: () => runCompletionIntroStep(),
      reset: () => resetCompletionActivity()
    }),
    'slide-32': createSlideDescriptor('slide-32', {
      init: () => resetCompletionActivity(),
      step: () => runPromptFixStep(),
      reset: () => resetCompletionActivity()
    }),
    'slide-32b': createSlideDescriptor('slide-32b', {
      init: () => initLiveFailureDemo(),
      reset: () => resetLiveFailureDemo()
    })
  });

  state.nav.slides.forEach((slideEl) => {
    const id = slideEl.id;
    descriptors.push(interactive[id] || createSlideDescriptor(id));
  });
  return descriptors;
}

function registerSlides() {
  state.registry.order = defineSlides();
  state.registry.byId = {};
  state.registry.order.forEach((descriptor) => {
    state.registry.byId[descriptor.id] = descriptor;
  });
}

function getSlideDescriptorById(id) {
  return state.registry.byId[id] || createSlideDescriptor(id);
}
