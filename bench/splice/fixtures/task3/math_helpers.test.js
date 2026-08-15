// Tests for math_helpers. Fails until double exists.
const test = require("node:test");
const assert = require("node:assert");
const { double } = require("./math_helpers");

test("double multiplies by two", () => {
  assert.strictEqual(double(4), 8);
  assert.strictEqual(double(0), 0);
});
