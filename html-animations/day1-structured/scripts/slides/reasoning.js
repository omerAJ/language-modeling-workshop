// ══════════════════════════════════════
//  Reasoning-emergence slides
// ══════════════════════════════════════
function revealReasoningPressure() {
  const grid = document.getElementById('reasoningPressureGrid');
  const quote = document.getElementById('reasoningEmergenceQuote');
  if (grid) grid.classList.add('revealed');
  if (quote) quote.classList.add('revealed');
}

function resetReasoningPressureSlide() {
  const grid = document.getElementById('reasoningPressureGrid');
  const quote = document.getElementById('reasoningEmergenceQuote');
  if (grid) grid.classList.remove('revealed', 'settled');
  if (quote) quote.classList.remove('revealed', 'settled');
}

function runReasoningPressureStep() {
  const pressureGrid = document.getElementById('reasoningPressureGrid');
  if (pressureGrid && !pressureGrid.classList.contains('revealed')) {
    revealReasoningPressure();
    return true;
  }
  return false;
}

function revealDramaAnswer() {
  const blank = document.getElementById('dramaKillerBlank');
  const why = document.getElementById('dramaWhy');
  const result = document.getElementById('dramaResult');
  if (blank && !blank.classList.contains('show')) {
    blank.textContent = blank.dataset.answer;
    blank.classList.add('show');
  }
  if (why) why.classList.add('revealed');
  if (result) {
    result.className = 'callout success';
    result.style.maxWidth = '';
    result.innerHTML = '<span class="icon">✓</span><span>Best next-token continuation here: <strong>Rukhsana</strong>, based on timing + access constraints.</span>';
  }
}

function chooseDramaSuspect(choice, btn) {
  const buttons = document.querySelectorAll('#dramaChoices .drama-choice');
  buttons.forEach((b) => b.classList.remove('correct', 'wrong'));
  if (choice === 'Rukhsana') btn.classList.add('correct');
  else btn.classList.add('wrong');

  const result = document.getElementById('dramaResult');
  if (result) {
    const isCorrect = choice === 'Rukhsana';
    result.className = isCorrect ? 'callout success' : 'callout warn';
    result.style.maxWidth = '';
    result.innerHTML = isCorrect
      ? '<span class="icon">✓</span><span>Consistent choice. Timing and pantry-access constraints point to <strong>Rukhsana</strong>.</span>'
      : '<span class="icon">!</span><span>This choice conflicts with the clue chain. Re-check who could act between 10:20 and 10:30.</span>';
  }

  if (choice === 'Rukhsana') {
    revealDramaAnswer();
  }
}

function resetDramaSlide() {
  const blank = document.getElementById('dramaKillerBlank');
  const why = document.getElementById('dramaWhy');
  const result = document.getElementById('dramaResult');
  const buttons = document.querySelectorAll('#dramaChoices .drama-choice');
  if (blank) {
    blank.textContent = '___';
    blank.classList.remove('show');
  }
  if (why) why.classList.remove('revealed', 'settled');
  buttons.forEach((b) => b.classList.remove('correct', 'wrong'));
  if (result) {
    result.className = 'callout warn';
    result.style.maxWidth = '';
    result.innerHTML = INITIAL_DRAMA_RESULT_HTML;
  }
}

function runDramaSlideStep() {
  const blank = document.getElementById('dramaKillerBlank');
  if (blank && !blank.classList.contains('show')) {
    revealDramaAnswer();
    return true;
  }
  return false;
}
