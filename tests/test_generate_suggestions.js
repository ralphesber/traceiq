/**
 * Unit tests for generateSuggestions()
 * Run with: node tests/test_generate_suggestions.js
 *
 * Inline copy of the function under test (matches index.html implementation).
 */

function generateSuggestions(overviewData) {
  const suggestions = [];
  if (overviewData) {
    const stats = overviewData.summary || {};
    const cmp = overviewData.comparison || {};
    const latestExp = overviewData.latest_experiment;

    const errRate = stats.error_rate ?? 0;
    const countDelta = cmp.count_delta ?? 0;
    const errRateDelta = cmp.error_rate_delta ?? 0;

    if (errRate > 5) {
      suggestions.push('What do failing traces have in common?');
    }
    if (countDelta < -20) {
      suggestions.push('Why did trace volume drop significantly this week?');
    }
    if (errRateDelta > 20) {
      suggestions.push('Why did the error rate spike compared to last week?');
    }
    if (latestExp && latestExp.scores) {
      const entries = Object.entries(latestExp.scores).filter(([, v]) => typeof v === 'number');
      if (entries.length > 0) {
        const [worstMetric] = entries.sort((a, b) => a[1] - b[1])[0];
        const expName = latestExp.name || latestExp.experiment_name || 'the experiment';
        suggestions.push(`Why is ${worstMetric} scoring low in ${expName}?`);
      }
    }
  }

  const generics = [
    'What patterns exist in low-quality outputs?',
    'Are there systematic differences between high and low scoring responses?',
  ];
  for (const g of generics) {
    if (suggestions.length >= 2) break;
    if (!suggestions.includes(g)) suggestions.push(g);
  }
  if (!suggestions.some(s => generics.includes(s))) {
    suggestions.push(generics[0]);
  }

  return suggestions.slice(0, 4).map(s => s.length > 60 ? s.slice(0, 57) + '…' : s);
}

// ── Test runner ─────────────────────────────────────────────────────
let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`  ✓ ${message}`);
    passed++;
  } else {
    console.error(`  ✗ FAIL: ${message}`);
    failed++;
  }
}

function test(name, fn) {
  console.log(`\n${name}`);
  fn();
}

// ── Tests ────────────────────────────────────────────────────────────

test('returns "What do failing traces have in common?" when error_rate > 5%', () => {
  const data = { summary: { error_rate: 8 }, comparison: {}, latest_experiment: null };
  const result = generateSuggestions(data);
  assert(result.includes('What do failing traces have in common?'), 'high error rate suggestion present');
});

test('does NOT return error suggestion when error_rate <= 5%', () => {
  const data = { summary: { error_rate: 3 }, comparison: {}, latest_experiment: null };
  const result = generateSuggestions(data);
  assert(!result.includes('What do failing traces have in common?'), 'no error suggestion for low error rate');
});

test('returns volume drop suggestion when count_delta < -20', () => {
  const data = { summary: { error_rate: 0 }, comparison: { count_delta: -35 }, latest_experiment: null };
  const result = generateSuggestions(data);
  assert(result.includes('Why did trace volume drop significantly this week?'), 'volume drop suggestion present');
});

test('returns error spike suggestion when error_rate_delta > 20', () => {
  const data = { summary: { error_rate: 0 }, comparison: { error_rate_delta: 30 }, latest_experiment: null };
  const result = generateSuggestions(data);
  assert(result.includes('Why did the error rate spike compared to last week?'), 'error spike suggestion present');
});

test('returns experiment suggestion when latest_experiment has scores', () => {
  const data = {
    summary: { error_rate: 0 },
    comparison: {},
    latest_experiment: {
      name: 'my_experiment',
      scores: { marks_exact_match: 0.3, coherence: 0.8 },
    },
  };
  const result = generateSuggestions(data);
  assert(result.some(s => s.includes('marks_exact_match') && s.includes('my_experiment')), 'experiment metric suggestion present');
});

test('always returns at least 2 suggestions', () => {
  // No contextual signals at all
  const data = { summary: { error_rate: 0 }, comparison: {}, latest_experiment: null };
  const result = generateSuggestions(data);
  assert(result.length >= 2, `got ${result.length} suggestions (expected >= 2)`);
});

test('always returns at least 2 suggestions when called with null', () => {
  const result = generateSuggestions(null);
  assert(result.length >= 2, `got ${result.length} suggestions (expected >= 2)`);
});

test('never returns more than 4 suggestions', () => {
  // Trigger every contextual signal
  const data = {
    summary: { error_rate: 10 },
    comparison: { count_delta: -50, error_rate_delta: 40 },
    latest_experiment: { name: 'exp1', scores: { accuracy: 0.2, recall: 0.5 } },
  };
  const result = generateSuggestions(data);
  assert(result.length <= 4, `got ${result.length} suggestions (expected <= 4)`);
});

test('truncates suggestions longer than 60 chars', () => {
  // Craft a long experiment name to force truncation
  const data = {
    summary: { error_rate: 0 },
    comparison: {},
    latest_experiment: {
      name: 'a_very_long_experiment_name_that_will_cause_overflow_in_the_chip',
      scores: { marks_exact_match: 0.1 },
    },
  };
  const result = generateSuggestions(data);
  result.forEach(s => {
    assert(s.length <= 60, `suggestion is within 60 chars: "${s}" (${s.length})`);
  });
});

test('always includes at least one generic suggestion', () => {
  const generics = [
    'What patterns exist in low-quality outputs?',
    'Are there systematic differences between high and low scoring responses?',
  ];
  const data = { summary: { error_rate: 8 }, comparison: { count_delta: -30, error_rate_delta: 25 }, latest_experiment: null };
  const result = generateSuggestions(data);
  assert(result.some(s => generics.includes(s)), 'at least one generic suggestion present');
});

// ── Summary ──────────────────────────────────────────────────────────
console.log(`\n${'─'.repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
