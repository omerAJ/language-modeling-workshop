/* Slide 27: Attention Mixes, FFN Transforms.
   Pure autostep — no JS step controller needed.
   Keep init/reset stubs for the registry. */

function initFfnCombinedSlide() {
  ffnCombinedState.initialized = true;
}

function runFfnCombinedStep() {
  return false;
}

function resetFfnCombinedSlide() {
  ffnCombinedState.initialized = false;
}
