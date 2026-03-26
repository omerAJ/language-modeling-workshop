/* Slide 26: Give Every Position a Fingerprint.
   Pure autostep — no JS step controller needed.
   Keep init/reset stubs for the registry. */

function initPositionSignalSlide() {
  positionSignalState.initialized = true;
}

function runPositionSignalStep() {
  return false;
}

function resetPositionSignalSlide() {
  positionSignalState.initialized = false;
}
