// Text utilities for the passable functionality task.

function firstWord(s) {
  const match = s.match(/\S+/);
  return match[0];
}

module.exports = { firstWord };
