// ══════════════════════════════════════
//  Inference demo + training loop
// ══════════════════════════════════════
function appendInferenceBlank() {
  const row = document.getElementById('inferenceDemo');
  if (!row || document.getElementById('infCurrent')) return;

  const newBlank = document.createElement('span');
  newBlank.className = 'tok blank';
  newBlank.id = 'infCurrent';
  newBlank.textContent = 'next?';
  row.appendChild(newBlank);
}

function clearInferenceBlankTimer() {
  if (!inferenceState.blankTimer) return;
  clearTrackedTimeout(inferenceState.blankTimer);
  inferenceState.blankTimer = null;
}

function scheduleInferenceBlank() {
  clearInferenceBlankTimer();
  inferenceState.pendingBlank = true;
  inferenceState.blankTimer = setTrackedTimeout(function() {
    inferenceState.blankTimer = null;
    inferenceState.pendingBlank = false;
    appendInferenceBlank();
  }, 350);
}

function resumeInferenceBlank() {
  if (!inferenceState.pendingBlank) return;
  if (document.getElementById('infCurrent')) {
    inferenceState.pendingBlank = false;
    return;
  }
  scheduleInferenceBlank();
}

function resetInferenceDemo() {
  const row = document.getElementById('inferenceDemo');
  clearInferenceBlankTimer();
  inferenceState.step = 0;
  inferenceState.pendingBlank = false;

  if (row) {
    row.innerHTML = [
      '<span class="tok input">The</span>',
      '<span class="tok input">capital</span>',
      '<span class="tok input">of</span>',
      '<span class="tok input">Pakistan</span>',
      '<span class="tok input">is</span>',
      '<span class="tok blank" id="infCurrent">next?</span>'
    ].join('');
  }
}

function runInferenceSlideStep() {
  if (inferenceState.pendingBlank) {
    clearInferenceBlankTimer();
    inferenceState.pendingBlank = false;
    appendInferenceBlank();
    return true;
  }
  if (inferenceState.step >= INF_TOKENS.length) return false;
  if (!document.getElementById('infCurrent')) return false;
  animateInference();
  return true;
}

function animateInference() {
  if (inferenceState.step >= INF_TOKENS.length) return;

  const row = document.getElementById('inferenceDemo');
  const cur = document.getElementById('infCurrent');
  if (!row || !cur) return;

  cur.removeAttribute('id');
  cur.classList.remove('blank');
  cur.style.cursor = 'default';

  const tok = INF_TOKENS[inferenceState.step];
  cur.textContent = tok;

  if (tok === '⟨EOS⟩') {
    cur.classList.add('done');
  } else {
    cur.classList.add('target');
    cur.style.borderStyle = 'solid';
  }

  inferenceState.step += 1;

  if (inferenceState.step < INF_TOKENS.length) {
    scheduleInferenceBlank();
  } else {
    clearInferenceBlankTimer();
    inferenceState.pendingBlank = false;
  }
}

function runTrainingLoopStep() {
  if (trainingState.phase >= 5) return false;
  trainStep(trainingState.phase + 1);
  return true;
}

function trainStep(phase) {
  trainingState.phase = phase;

  for (var i = 1; i <= 5; i++) {
    var btn = document.getElementById('trainBtn' + i);
    btn.style.opacity = i <= phase ? '1' : '0.45';
    btn.style.borderColor = i === phase ? 'var(--cyan)' : '';
  }

  document.getElementById('trainFwd').style.opacity = phase >= 1 ? '1' : '0.25';
  document.getElementById('trainCompare').style.display = phase >= 2 ? '' : 'none';
  document.getElementById('trainLoss').style.display = phase >= 3 ? '' : 'none';
  document.getElementById('trainBackprop').style.display = phase >= 4 ? '' : 'none';
  document.getElementById('trainUpdate').style.display = phase >= 5 ? '' : 'none';

  var tfBox = document.getElementById('trainTransformer');
  if (phase === 4) {
    tfBox.style.borderColor = 'rgba(251,146,60,0.6)';
    tfBox.style.boxShadow = '0 0 8px rgba(251,146,60,0.2)';
  } else if (phase === 5) {
    tfBox.style.borderColor = 'var(--orange)';
    tfBox.style.boxShadow = '0 0 14px rgba(251,146,60,0.35)';
  } else {
    tfBox.style.borderColor = '';
    tfBox.style.boxShadow = '';
  }

  var explainDiv = document.getElementById('trainExplain');
  var explainText = document.getElementById('trainExplainText');
  explainDiv.style.display = '';
  explainText.innerHTML = TRAIN_EXPLANATIONS[phase];
  typesetMath(explainDiv);
}

function resetTrainLoop() {
  trainingState.phase = 0;
  document.getElementById('trainFwd').style.opacity = '0.25';
  document.getElementById('trainCompare').style.display = 'none';
  document.getElementById('trainLoss').style.display = 'none';
  document.getElementById('trainBackprop').style.display = 'none';
  document.getElementById('trainUpdate').style.display = 'none';
  document.getElementById('trainExplain').style.display = '';
  document.getElementById('trainExplainText').innerHTML = TRAIN_INTRO_HTML;
  var tfBox = document.getElementById('trainTransformer');
  tfBox.style.borderColor = '';
  tfBox.style.boxShadow = '';
  for (var i = 1; i <= 5; i++) {
    var btn = document.getElementById('trainBtn' + i);
    btn.style.opacity = '';
    btn.style.borderColor = '';
  }
}
