#!/usr/bin/env bash
# Fail if anything that must not be published made it into the tree: non-English
# text, credentials, absolute paths from a development machine, or leftover
# internal naming.
#
#     bash tools/check_release_hygiene.sh
#
# Runs over everything git would publish: tracked files plus untracked ones that
# .gitignore does not exclude. Exits non-zero if any category has hits.
#
# Patterns are written generically on purpose. Naming particular hosts, accounts
# or internal projects would publish the very list this script exists to keep out.
# To add your own, put one grep pattern per line in a file and point
# HYGIENE_EXTRA_PATTERNS at it; the file is read, never committed.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

fail=0

# Files allowed to mention otherwise-banned strings: this script, and the
# provenance record that has to name the upstream repository.
# Exempt: this script, whose patterns would match themselves; the provenance record,
# which has to name the upstream repository; and .gitignore, whose root-anchored
# patterns such as "/data/" read as absolute paths to the check below.
allow='^(tools/check_release_hygiene\.sh|third_party/qwen_vl_finetune/PROVENANCE\.md|\.gitignore)$'

files() {
  git ls-files --cached --others --exclude-standard \
    | grep -Ev "$allow" \
    | grep -Ev '\.(png|jpg|jpeg|gif|pdf|safetensors|bin|parquet)$'
}

# Perl-compatible patterns so that \x{...} ranges and lookarounds behave the same
# way regardless of the caller's locale.
check() {
  local name="$1" pattern="$2"
  local hits
  hits=$(files | xargs -r grep -nPI "$pattern" 2>/dev/null || true)
  if [[ -n "$hits" ]]; then
    echo "FAIL  ${name}"
    echo "$hits" | sed 's/^/      /'
    fail=1
  else
    echo "ok    ${name}"
  fi
}

echo "== release hygiene =="

# Non-English source text: Hangul syllables and Jamo, and CJK ideographs.
check "no CJK or Hangul"        '[\x{AC00}-\x{D7A3}\x{1100}-\x{11FF}\x{3130}-\x{318F}\x{4E00}-\x{9FFF}\x{3040}-\x{30FF}]'

# Credentials.
check "no API keys or tokens"   '\b(hf|sk|ghp|gho|github_pat)_[A-Za-z0-9]{20,}|wandb_v1_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}'
check "no assignments of one"   '(?i)\b(api[_-]?key|secret|password|token)\s*[:=]\s*["'"'"']?[A-Za-z0-9/+_-]{16,}'

# Absolute paths that exist only on one machine: a path rooted at a directory a
# published file has no business naming. The path must start a token, which keeps
# URLs and relative paths such as "<run>/global_step_500" out, and the exclusion
# list covers the standard filesystem.
check "no absolute local paths" '(?:^|(?<=[\s"\x27=(]))/(?!usr/|bin/|etc/|tmp/|dev/|opt/|proc/|sys/|var/|lib/|lib64/|sbin/|srv/|mnt/|media/|root/)[a-z][a-z0-9_.-]*/[A-Za-z0-9_.-]+'
check "no home directories"     '/home/[a-z][a-z0-9_.-]*|/Users/[A-Za-z]'

# Machine and network identifiers. Loopback is fine: single-node training uses it.
check "no IP addresses"         '(?<![\w.])(?!127\.0\.0\.1|0\.0\.0\.0|255\.)(?:\d{1,3}\.){3}\d{1,3}(?![\w.])'
check "no private hostnames"    '(?i)\b[a-z0-9-]+\.(?:local|internal|corp|lan)\b'

# Internal naming that must not survive the rename. Version suffixes are matched
# without a leading \b because "_" is itself a word character.
check "no version suffixes"     '_v\d{1,2}(?![\d\w])|\bv\d{1,2}_'

if [[ -n "${HYGIENE_EXTRA_PATTERNS:-}" ]]; then
  if [[ -r "$HYGIENE_EXTRA_PATTERNS" ]]; then
    n=0
    while IFS= read -r pattern; do
      [[ -z "$pattern" || "$pattern" == \#* ]] && continue
      n=$((n + 1))
      check "extra pattern $n" "$pattern"
    done < "$HYGIENE_EXTRA_PATTERNS"
  else
    echo "WARN  HYGIENE_EXTRA_PATTERNS is set but not readable: $HYGIENE_EXTRA_PATTERNS"
  fi
fi

echo
if [[ $fail -eq 0 ]]; then
  echo "PASS: no findings"
else
  echo "FAILED: fix the findings above before publishing"
fi
exit $fail
