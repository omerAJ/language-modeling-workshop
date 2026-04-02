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
  const interactive = {
    'slide-3': createSlideDescriptor('slide-3', {
      init: () => initArchitectureOverviewSlide(),
      step: () => runArchitectureOverviewStep(),
      reset: () => resetArchitectureOverviewSlide()
    }),
    'slide-16': createSlideDescriptor('slide-16', {
      init: () => {
        initProjectionSlide();
        drawProjection();
      },
      step: (slideEl) => runProjectionLensStep(slideEl),
      reset: () => resetProjectionSlide()
    }),
    'slide-18': createSlideDescriptor('slide-18', {
      init: () => {
        initAttentionIntroSlide();
        setAttentionIntroStep(0);
      },
      step: () => runAttentionIntroStep(),
      reset: () => resetAttentionIntroSlide()
    }),
    'slide-19': createSlideDescriptor('slide-19', {
      init: () => {
        initAttentionP1Slide();
        setAttentionP1Step(0);
      },
      step: () => runAttentionP1Step(),
      reset: () => resetAttentionP1Slide()
    }),
    'slide-20': createSlideDescriptor('slide-20', {
      init: () => {
        initAttentionQkvSlide();
        setAttentionQkvStep(0);
      },
      step: () => runAttentionQkvStep(),
      reset: () => resetAttentionQkvSlide()
    }),
    'slide-21': createSlideDescriptor('slide-21', {
      init: () => {
        initAttentionWeightsSlide();
        setAttentionWeightsStep(0);
      },
      step: () => runAttentionWeightsStep(),
      reset: () => resetAttentionWeightsSlide()
    }),
    'slide-22': createSlideDescriptor('slide-22', {
      init: () => {
        initAttentionStep4Slide();
        setAttentionStep4Step(0);
      },
      step: () => runAttentionStep4Step(),
      reset: () => resetAttentionStep4Slide()
    }),
    'slide-23': createSlideDescriptor('slide-23', {
      init: () => {
        initAttentionMatrixSlide();
        setAttentionMatrixStep(0);
      },
      step: () => runAttentionMatrixStep(),
      reset: () => resetAttentionMatrixSlide()
    }),
    'slide-24': createSlideDescriptor('slide-24', {
      init: () => {
        initAttentionMultiHeadSlide();
        setAttentionMultiHeadStep(0);
      },
      step: () => runAttentionMultiHeadStep(),
      reset: () => resetAttentionMultiHeadSlide()
    }),
    'slide-25': createSlideDescriptor('slide-25', {
      init: () => initOrderProblemSlide(),
      reset: () => resetOrderProblemSlide()
    }),
    'slide-26': createSlideDescriptor('slide-26', {
      init: () => initPositionSignalSlide(),
      step: () => runPositionSignalStep(),
      reset: () => resetPositionSignalSlide()
    }),
    'slide-27': createSlideDescriptor('slide-27', {
      init: () => initFfnCombinedSlide(),
      reset: () => resetFfnCombinedSlide()
    }),
    'slide-28': createSlideDescriptor('slide-28', {
      init: () => initGptBlockSlide(),
      step: () => runGptBlockStep(),
      reset: () => resetGptBlockSlide()
    }),
    'slide-32': createSlideDescriptor('slide-32', {
      init: () => initOutputHeadSlide(),
      step: () => runOutputHeadStep(),
      reset: () => resetOutputHeadSlide()
    }),
    'slide-33': createSlideDescriptor('slide-33', {
      init: () => initGenerationSlide(),
      step: () => runGenerationStep(),
      reset: () => resetGenerationSlide()
    })
  };

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
