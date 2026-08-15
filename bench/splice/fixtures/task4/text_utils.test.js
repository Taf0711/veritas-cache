// Tests for text_utils. The empty input test fails until the task is done.
const test = require("node:test");
const assert = require("node:assert");
const { firstWord } = require("./text_utils");

test("first word of normal text", () => {
  assert.strictEqual(firstWord("hi there"), "hi");
});

test("first word of empty text", () => {
  assert.strictEqual(firstWord(""), "");
});
