            // ══════════════════════════════════════
            //  NTP game
            // ══════════════════════════════════════
            function revealNTP(el) {
  if (el.classList.contains('show')) return;
  el.textContent = el.dataset.answer;
  el.classList.add('show');
}
function revealNTPConcept(btn) {
  const sentence = btn.closest('.ntp-sentence');
  const tag = sentence ? sentence.querySelector('.ntp-tag') : null;
  if (tag) tag.classList.add('show');
  btn.style.display = 'none';
}
function revealAllNTP() {
  $$('.ntp-blank').forEach(el => {
    el.textContent = el.dataset.answer;
    el.classList.add('show');
  });
  $$('.ntp-concept-btn').forEach(btn => { btn.style.display = 'none'; });
  $$('.ntp-tag').forEach(tag => tag.classList.add('show'));
}

            function runNtpSlideStep() {
              var rows = document.querySelectorAll('#slide-16 .ntp-sentence');
              for (var r = 0; r < rows.length; r++) {
                var blank = rows[r].querySelector('.ntp-blank');
                var conceptBtn = rows[r].querySelector('.ntp-concept-btn');

                if (blank && !blank.classList.contains('show')) {
                  revealNTP(blank);
                  return true;
                }
                if (conceptBtn && conceptBtn.style.display !== 'none') {
                  revealNTPConcept(conceptBtn);
                  return true;
                }
              }
              return false;
            }

            // ══════════════════════════════════════
            //  Bar chart builder (for loss deep-dive)
            // ══════════════════════════════════════
            function buildBars(container, data, type) {
  const el = typeof container === 'string' ? $(container) : container;
  if (!el) return;
  el.innerHTML = '';
  data.forEach((d, i) => {
    const row = document.createElement('div');
    row.className = 'bar-row';

    const wordEl = document.createElement('div');
    wordEl.className = 'bar-word';
    wordEl.textContent = d.word;
    if (d.word === 'Islamabad') wordEl.style.color = 'var(--green)';

    const track = document.createElement('div');
    track.className = 'bar-track';

    const fill = document.createElement('div');
    fill.className = 'bar-fill ' + type;
    const pctWidth = d.value > 0 ? Math.max(d.value * 100, 1.5) : 0;
    fill.style.width = '0%';
    setTimeout(() => { fill.style.width = pctWidth + '%'; }, 80 + i * 50);

    track.appendChild(fill);

    const pctEl = document.createElement('div');
    pctEl.className = 'bar-pct';
    pctEl.textContent = `\\(${(d.value * 100).toFixed(1)}\\%\\)`;

    row.appendChild(wordEl);
    row.appendChild(track);
    row.appendChild(pctEl);
    el.appendChild(row);
  });
  // Bars can be inserted after slide-enter (e.g., delayed "after update" chart),
  // so explicitly typeset newly injected LaTeX labels here.
  typesetMath(el);
}

function predictedData() { return VOCAB.map(v => ({ word: v.word, value: v.predicted })); }
function targetData()    { return VOCAB.map(v => ({ word: v.word, value: v.target })); }
function afterData()     { return VOCAB.map(v => ({ word: v.word, value: v.after })); }

            function initPredictedChartSlide() {
              buildBars('#predictedChart', predictedData(), 'predicted');
            }

            function initTargetComparisonSlide() {
              buildBars('#dualPredicted', predictedData(), 'predicted');
              buildBars('#dualTarget', targetData(), 'target-fill');
              const pct = VOCAB[0].predicted;
              $('#correctPctText').textContent = `\\(${(pct * 100).toFixed(0)}\\%\\)`;
            }

            function initLossComputationSlide() {
              const p = VOCAB[0].predicted;
              const loss = -Math.log(p);
              const pText = p.toFixed(2);
              const lossText = loss.toFixed(2);
              $('#lossFormula').textContent = `\\[\\mathcal{L} = -\\log(P_{\\text{model}}(\\text{Islamabad})) = -\\log(${pText}) = ${lossText}\\]`;
              $('#lossProbDisplay').textContent = `\\(${pText}\\)`;
              $('#lossValueDisplay').textContent = `\\(${lossText}\\)`;
              const meter = $('#lossMeter');
              meter.style.width = '0%';
              setTimeout(() => {
                meter.style.width = Math.min(loss / 5 * 100, 100) + '%';
              }, 200);
            }

            function initAfterUpdateSlide() {
              buildBars('#beforeBars', predictedData(), 'predicted');
              setTimeout(() => {
                buildBars('#afterBars', afterData(), 'target-fill');
              }, 300);
              const afterLoss = -Math.log(VOCAB[0].after);
              $('#lossAfterVal').textContent = `\\(${afterLoss.toFixed(2)}\\)`;
              $('#afterCorrectPct').textContent = `\\(${(VOCAB[0].after * 100).toFixed(0)}\\%\\)`;
              $('#afterLossText').textContent = `\\(${afterLoss.toFixed(2)}\\)`;
            }
