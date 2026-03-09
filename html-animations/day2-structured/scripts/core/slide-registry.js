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
      step: (slideEl) => runAttentionIntroStep(slideEl),
      reset: () => resetAttentionIntroSlide()
    }),
    'slide-19': createSlideDescriptor('slide-19', {
      init: () => {
        initAttentionQkvSlide();
        setAttentionQkvStep(0);
      },
      step: (slideEl) => runAttentionQkvStep(slideEl),
      reset: () => resetAttentionQkvSlide()
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
