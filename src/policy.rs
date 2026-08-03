// Define the decision that a policy returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Hit,
    Miss,
}

// Define the policy hooks used by the replay engine.
pub trait Policy {
    fn decide(&mut self, neighbor: Option<(usize, f32)>) -> Decision;
    fn should_insert(&self) -> bool;
    fn observe(&mut self, _entry: usize, _sim: f32, _correct: bool) {}
}

// Apply one global cosine threshold.
#[derive(Debug, Clone, Copy)]
pub struct StaticPolicy {
    pub threshold: f32,
}

impl Policy for StaticPolicy {
    fn decide(&mut self, neighbor: Option<(usize, f32)>) -> Decision {
        match neighbor {
            Some((_, similarity)) if similarity >= self.threshold => Decision::Hit,
            _ => Decision::Miss,
        }
    }

    fn should_insert(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_policy_hits_at_threshold_boundary() {
        let mut policy = StaticPolicy { threshold: 0.8 };
        assert_eq!(policy.decide(Some((2, 0.8))), Decision::Hit);
        assert_eq!(policy.decide(Some((2, 0.7999))), Decision::Miss);
        assert_eq!(policy.decide(None), Decision::Miss);
        assert!(policy.should_insert());
    }
}
