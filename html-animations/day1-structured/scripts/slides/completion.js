// ══════════════════════════════════════
//  Completion wrapper activity
// ══════════════════════════════════════
function revealDumbResponse() {
  const reveal = document.getElementById('dumbResponseReveal');
  if (reveal) reveal.classList.add('revealed');
}

function clearCompletionBridgeTimer() {
  if (!completionState.bridgeTimer) return;
  clearTrackedTimeout(completionState.bridgeTimer);
  completionState.bridgeTimer = null;
}

function revealCompletionBridge() {
  var takeaway = document.getElementById('fixTakeaway');
  var bridge = document.getElementById('fixBridge');
  if (takeaway) takeaway.classList.add('revealed');
  if (bridge) bridge.classList.add('revealed');
}

function scheduleCompletionBridgeReveal() {
  clearCompletionBridgeTimer();
  completionState.pendingBridge = true;
  completionState.bridgeTimer = setTrackedTimeout(function() {
    completionState.bridgeTimer = null;
    completionState.pendingBridge = false;
    revealCompletionBridge();
  }, 400);
}

function resumeCompletionBridgeReveal() {
  if (!completionState.pendingBridge) return;
  scheduleCompletionBridgeReveal();
}

function resetCompletionState() {
  completionState.fixesRevealed = { chat: false, math: false, code: false };
  completionState.pendingBridge = false;
  clearCompletionBridgeTimer();
}

function revealFix(which, autoBridge) {
  if (autoBridge === undefined) autoBridge = true;
  var el = document.getElementById('fix' + which.charAt(0).toUpperCase() + which.slice(1));
  if (el) el.classList.add('revealed');
  completionState.fixesRevealed[which] = true;
  if (
    autoBridge &&
    completionState.fixesRevealed.chat &&
    completionState.fixesRevealed.math &&
    completionState.fixesRevealed.code
  ) {
    scheduleCompletionBridgeReveal();
  }
}

function resetCompletionActivity() {
  resetCompletionState();
  ['dumbResponseReveal', 'completionPhase2', 'fixChat', 'fixMath', 'fixCode', 'fixTakeaway', 'fixBridge'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.classList.remove('revealed', 'settled');
  });
}

function runCompletionIntroStep() {
  var reveal = document.getElementById('dumbResponseReveal');
  if (reveal && !reveal.classList.contains('revealed')) {
    revealDumbResponse();
    return true;
  }
  return false;
}

function runPromptFixStep() {
  var phase2 = document.getElementById('completionPhase2');
  if (phase2 && !phase2.classList.contains('revealed')) {
    phase2.classList.add('revealed');
    return true;
  }

  var fixChat = document.getElementById('fixChat');
  if (fixChat && !fixChat.classList.contains('revealed')) {
    revealFix('chat', false);
    return true;
  }
  var fixMath = document.getElementById('fixMath');
  if (fixMath && !fixMath.classList.contains('revealed')) {
    revealFix('math', false);
    return true;
  }
  var fixCode = document.getElementById('fixCode');
  if (fixCode && !fixCode.classList.contains('revealed')) {
    revealFix('code', false);
    return true;
  }

  var takeaway = document.getElementById('fixTakeaway');
  if (takeaway && !takeaway.classList.contains('revealed')) {
    takeaway.classList.add('revealed');
    return true;
  }

  var bridge = document.getElementById('fixBridge');
  if (bridge && !bridge.classList.contains('revealed')) {
    bridge.classList.add('revealed');
    return true;
  }
  return false;
}
