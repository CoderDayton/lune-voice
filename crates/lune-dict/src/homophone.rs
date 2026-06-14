//! Personalized homophone resolution.
//!
//! Curated confusion sets plus the per-user n-gram LM (`lune-ngram`,
//! trained on the user's own transcripts) disambiguate homophones the
//! acoustic model cannot — "there"/"their"/"they're", "to"/"too"/"two".
//!
//! This is deliberately *not* open-vocabulary candidate correction. The
//! earlier weighted-edit-distance approach was abandoned because generating
//! candidates over a 200k lexicon is noisy ("too hard to know what to correct
//! to"). Here candidates come from a fixed, high-precision table, so the only
//! question left is *which member fits this context* — exactly what the
//! n-gram answers.
//!
//! ## Safety
//!
//! Rewriting an ASR word is risky: over-correction degrades quality more than
//! the errors it fixes. So a word is only rewritten when ALL hold:
//!   1. it belongs to a confusion set;
//!   2. the n-gram's confidence MARGIN for its best member over the observed
//!      word exceeds [`ResolverPolicy::MarginGate`]'s `tau` (correct only when
//!      confident — magnitude, not just ranking);
//!   3. the 2-word left context has real training support
//!      ([`ResolverConfig::min_context_support`]), so the decision reflects
//!      context rather than global word frequency;
//!   4. the word is not acoustically broken
//!      ([`ResolverConfig::max_word_risk`], reusing the HWCE risk signal).
//!
//! Tuning experiments (`benchmarks/homophone_tune.rs`) show the margin gate
//! roughly halves homophone error relative to the ungated rerank-argmax policy
//! while cutting over-correction ~3×. With no learnable per-user signal the
//! resolver is net-harmful, so it is intended to ship opt-in / auto-disabled
//! per set — never applied blindly.

use std::collections::HashMap;

use lune_ngram::{AsrRerankWeights, NGramModel, tokenize};

/// A set of mutually-confusable surface forms.
///
/// Members MUST tokenize to distinct tokens — the n-gram cannot separate
/// forms that normalize equal (e.g. "it's"/"its", "we're"/"were"), so those
/// pairs are intentionally absent. Enforced by `confusion_sets_are_distinct`.
#[derive(Debug, Clone, Copy)]
pub struct ConfusionSet {
    pub members: &'static [&'static str],
}

/// Curated English homophone / near-homophone sets that survive tokenization
/// as distinct tokens. High-frequency function words (to, for, no, then) are
/// included but rely on the margin + context-support gates to stay safe; the
/// benchmark is what decides per-set thresholds.
pub static CONFUSION_SETS: &[ConfusionSet] = &[
    ConfusionSet {
        members: &["there", "their", "they're"],
    },
    ConfusionSet {
        members: &["to", "too", "two"],
    },
    ConfusionSet {
        members: &["your", "you're"],
    },
    ConfusionSet {
        members: &["no", "know"],
    },
    ConfusionSet {
        members: &["than", "then"],
    },
    ConfusionSet {
        members: &["here", "hear"],
    },
    ConfusionSet {
        members: &["write", "right"],
    },
    ConfusionSet {
        members: &["buy", "by", "bye"],
    },
    ConfusionSet {
        members: &["one", "won"],
    },
    ConfusionSet {
        members: &["for", "four"],
    },
    ConfusionSet {
        members: &["weather", "whether"],
    },
    ConfusionSet {
        members: &["lose", "loose"],
    },
    ConfusionSet {
        members: &["accept", "except"],
    },
    ConfusionSet {
        members: &["affect", "effect"],
    },
    ConfusionSet {
        members: &["whose", "who's"],
    },
];

/// Which member-selection rule the resolver applies.
#[derive(Debug, Clone, Copy)]
pub enum ResolverPolicy {
    /// Never rewrite. The resolver is a no-op.
    Off,
    /// Rewrite to the argmax of the blended rerank score (lune-ngram's
    /// `rerank_asr_candidates`). Ungated — kept for benchmark comparison; it
    /// over-corrects because min-max normalization discards confidence on
    /// small candidate sets.
    RerankArgmax,
    /// Rewrite to the best n-gram member only when its avg-log-prob margin
    /// over the observed word exceeds `tau` (nats). The recommended policy.
    MarginGate { tau: f64 },
}

/// Resolver tunables. Defaults are the benchmark's recommended operating point.
#[derive(Debug, Clone, Copy)]
pub struct ResolverConfig {
    pub policy: ResolverPolicy,
    /// Minimum trigram-context support before the n-gram is trusted.
    pub min_context_support: usize,
    /// Skip resolution when the observed word's HWCE risk is at/above this.
    pub max_word_risk: f32,
    /// Blend weights used when `policy == RerankArgmax`.
    pub rerank_weights: AsrRerankWeights,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            policy: ResolverPolicy::MarginGate { tau: 1.5 },
            min_context_support: 1,
            max_word_risk: 0.85,
            rerank_weights: AsrRerankWeights::transcript_default(),
        }
    }
}

/// A chosen replacement for one observed word.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPick {
    /// Canonical member to substitute (lowercase surface form).
    pub to: String,
    /// n-gram avg-log-prob margin of `to` over the observed word.
    pub margin: f64,
}

/// A resolution applied at a specific word index in a token stream.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolution {
    pub index: usize,
    pub from: String,
    pub to: String,
    pub margin: f64,
}

/// Confusion-set-bounded, confidence-gated homophone resolver.
pub struct HomophoneResolver {
    config: ResolverConfig,
    /// Normalized member surface form -> index into [`CONFUSION_SETS`].
    index: HashMap<String, usize>,
}

impl Default for HomophoneResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl HomophoneResolver {
    /// Resolver with [`ResolverConfig::default`] (margin gate, tau = 1.5).
    pub fn new() -> Self {
        Self::with_config(ResolverConfig::default())
    }

    pub fn with_config(config: ResolverConfig) -> Self {
        let mut index = HashMap::new();
        for (set_idx, set) in CONFUSION_SETS.iter().enumerate() {
            for &member in set.members {
                index.insert(norm_match(member), set_idx);
            }
        }
        Self { config, index }
    }

    pub fn config(&self) -> &ResolverConfig {
        &self.config
    }

    fn set_for(&self, word_norm: &str) -> Option<&'static ConfusionSet> {
        self.index.get(word_norm).map(|&i| &CONFUSION_SETS[i])
    }

    /// Decide a replacement for a single observed word given its (already
    /// tokenized) left context and HWCE risk. `None` means keep the word.
    pub fn resolve_word(
        &self,
        model: &NGramModel,
        left_ctx: &[String],
        observed: &str,
        word_risk: f32,
    ) -> Option<ResolvedPick> {
        if matches!(self.config.policy, ResolverPolicy::Off) {
            return None;
        }
        if word_risk >= self.config.max_word_risk {
            return None;
        }
        let obs_norm = norm_match(observed);
        let set = self.set_for(&obs_norm)?;
        let prefix = left_ctx.join(" ");

        match self.config.policy {
            ResolverPolicy::Off => None,
            ResolverPolicy::MarginGate { tau } => {
                if model.context_support(left_ctx) < self.config.min_context_support {
                    return None;
                }
                let mut best: Option<(&'static str, f64)> = None;
                let mut obs_lp = f64::NEG_INFINITY;
                for &member in set.members {
                    let lp = model.score_candidate(&prefix, member).avg_log_prob;
                    if norm_match(member) == obs_norm {
                        obs_lp = lp;
                    }
                    if best.is_none_or(|(_, b)| lp > b) {
                        best = Some((member, lp));
                    }
                }
                let (best_member, best_lp) = best?;
                if norm_match(best_member) == obs_norm {
                    return None; // observed already the n-gram's pick
                }
                let margin = best_lp - obs_lp;
                (margin > tau).then(|| ResolvedPick {
                    to: best_member.to_owned(),
                    margin,
                })
            }
            ResolverPolicy::RerankArgmax => {
                let best = model
                    .best_asr_candidate(
                        &prefix,
                        observed,
                        set.members.iter().copied(),
                        self.config.rerank_weights,
                    )
                    .ok()??;
                if norm_match(&best.candidate) == obs_norm {
                    return None;
                }
                let obs_lp = model.score_candidate(&prefix, &obs_norm).avg_log_prob;
                Some(ResolvedPick {
                    margin: best.ngram_avg_log_prob - obs_lp,
                    to: best.candidate,
                })
            }
        }
    }

    /// Scan `words` left-to-right, resolving each homophone slot using the
    /// already-resolved prefix as context. `risks[i]` is word `i`'s HWCE risk
    /// (use `&[]` or shorter slice for "unknown" → treated as 0.0). Returns the
    /// applied resolutions in order; `words` itself is not mutated.
    pub fn resolve_words(
        &self,
        model: &NGramModel,
        words: &[&str],
        risks: &[f32],
    ) -> Vec<Resolution> {
        let mut out = Vec::new();
        let mut ctx: Vec<String> = Vec::with_capacity(words.len() + 2);
        for (i, &word) in words.iter().enumerate() {
            let risk = risks.get(i).copied().unwrap_or(0.0);
            let chosen = match self.resolve_word(model, &ctx, word, risk) {
                Some(pick) => {
                    let to = pick.to.clone();
                    out.push(Resolution {
                        index: i,
                        from: word.to_owned(),
                        to: pick.to,
                        margin: pick.margin,
                    });
                    to
                }
                None => word.to_owned(),
            };
            ctx.extend(tokenize(&chosen));
        }
        out
    }
}

/// Lowercase and trim surrounding punctuation, preserving an interior
/// apostrophe so "They're," matches the member "they're".
fn norm_match(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_ngram::ScoringConfig;

    fn trained_model() -> NGramModel {
        let mut m = NGramModel::new(ScoringConfig::trigram_default()).unwrap();
        let corpus = [
            "send their patch to the channel",
            "their patch is ready for review",
            "review their patch again",
            "i left the keys over there",
            "the build is over there now",
            "it is right over there",
        ];
        for _ in 0..30 {
            m.train_sentences(corpus);
        }
        m
    }

    #[test]
    fn confusion_sets_are_distinct() {
        for set in CONFUSION_SETS {
            let toks: Vec<String> = set.members.iter().map(|m| tokenize(m).join(" ")).collect();
            let uniq: std::collections::HashSet<&String> = toks.iter().collect();
            assert_eq!(
                uniq.len(),
                toks.len(),
                "confusion set {:?} has members that tokenize identically",
                set.members
            );
        }
    }

    #[test]
    fn resolves_their_after_send() {
        let m = trained_model();
        let r = HomophoneResolver::new();
        // ASR misheard "their" as "there" after "send".
        let res = r.resolve_words(&m, &["send", "there", "patch"], &[]);
        assert!(
            res.iter().any(|x| x.from == "there" && x.to == "their"),
            "expected there->their after 'send', got {res:?}"
        );
    }

    #[test]
    fn keeps_there_after_over() {
        let m = trained_model();
        let r = HomophoneResolver::new();
        // ASR misheard "there" as "their" after "keys over" (a context the
        // model has actually seen — the support gate requires real evidence).
        let res = r.resolve_words(&m, &["keys", "over", "their"], &[]);
        assert!(
            res.iter().any(|x| x.to == "there"),
            "expected their->there after 'keys over', got {res:?}"
        );
    }

    #[test]
    fn off_policy_never_resolves() {
        let m = trained_model();
        let r = HomophoneResolver::with_config(ResolverConfig {
            policy: ResolverPolicy::Off,
            ..ResolverConfig::default()
        });
        assert!(
            r.resolve_words(&m, &["send", "there", "patch"], &[])
                .is_empty()
        );
    }

    #[test]
    fn high_margin_threshold_suppresses_flips() {
        let m = trained_model();
        let r = HomophoneResolver::with_config(ResolverConfig {
            policy: ResolverPolicy::MarginGate { tau: 1000.0 },
            ..ResolverConfig::default()
        });
        assert!(
            r.resolve_words(&m, &["send", "there", "patch"], &[])
                .is_empty()
        );
    }

    #[test]
    fn broken_word_skipped() {
        let m = trained_model();
        let r = HomophoneResolver::new(); // max_word_risk 0.85
        // Same flip as resolves_their_after_send, but the word is flagged broken.
        let res = r.resolve_words(&m, &["send", "there", "patch"], &[0.0, 0.95, 0.0]);
        assert!(
            res.is_empty(),
            "broken word must be left untouched, got {res:?}"
        );
    }

    #[test]
    fn non_homophone_untouched() {
        let m = trained_model();
        let r = HomophoneResolver::new();
        assert!(
            r.resolve_words(&m, &["send", "the", "patch"], &[])
                .is_empty()
        );
    }
}
