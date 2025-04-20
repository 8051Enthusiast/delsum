use std::str::Chars;
#[derive(Debug)]
enum State {
    Whitespace,
    Key,
    Equal,
    Value,
    QuotedValue,
    QuoteEnd,
}

pub(crate) struct KeyValIter<'a> {
    slicc: Chars<'a>,
    stop: bool,
}
impl<'a> KeyValIter<'a> {
    pub fn new(s: &'a str) -> KeyValIter<'a> {
        KeyValIter {
            slicc: s.chars(),
            stop: false,
        }
    }
}

impl Iterator for KeyValIter<'_> {
    type Item = Result<(String, String), String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.stop {
            return None;
        }
        let mut current_val = String::default();
        let mut current_key = String::default();
        let mut current_state = State::Whitespace;
        let mut is_kv_end;
        loop {
            is_kv_end = false;
            let c = match self.slicc.next() {
                Some(x) => {
                    if x.is_whitespace() {
                        ' '
                    } else {
                        x
                    }
                }
                None => {
                    self.stop = true;
                    ' '
                }
            };
            current_state = match (current_state, c) {
                (State::QuotedValue, '"') => {
                    is_kv_end = true;
                    State::QuoteEnd
                }
                (State::Value, ' ') => {
                    is_kv_end = true;
                    State::Whitespace
                }
                (State::QuotedValue, _) => {
                    current_val.push(c);
                    State::QuotedValue
                }
                (State::Value, 'A'..='Z')
                | (State::Value, 'a'..='z')
                | (State::Value, '0'..='9')
                | (State::Equal, 'A'..='Z')
                | (State::Equal, 'a'..='z')
                | (State::Equal, '0'..='9') => {
                    current_val.push(c);
                    State::Value
                }
                (State::Key, '=') => State::Equal,
                (State::Key, 'A'..='Z')
                | (State::Key, 'a'..='z')
                | (State::Key, '_')
                | (State::Whitespace, 'A'..='Z')
                | (State::Whitespace, 'a'..='z')
                | (State::Whitespace, '_') => {
                    current_key.push(c.to_ascii_lowercase());
                    State::Key
                }
                (State::Equal, '"') => State::QuotedValue,
                (State::Whitespace, ' ') | (State::QuoteEnd, ' ') => State::Whitespace,
                _ => {
                    self.stop = true;
                    return Some(Err(current_key));
                }
            };
            if is_kv_end || self.stop {
                break;
            }
        }
        if is_kv_end {
            Some(Ok((current_key, current_val)))
        } else {
            // if self.stop
            match current_state {
                State::Whitespace => None,
                _ => Some(Err(current_key)),
            }
        }
    }
}
