pub const DEFAULT_BIGRAMS: &[(&str, &str, u32)] = &[
    // a + *
    ("a", "big", 900),
    ("a", "good", 1000),
    ("a", "great", 950),
    ("a", "little", 1100),
    ("a", "lot", 2000),
    ("a", "new", 1200),
    // able + *
    ("able", "to", 1400),
    // after + *
    ("after", "the", 1600),
    // all + *
    ("all", "of", 1800),
    // are + *
    ("are", "you", 1300),
    // as + *
    ("as", "much", 1500),
    ("as", "well", 1900),
    // at + *
    ("at", "least", 1700),
    ("at", "the", 2500),
    // based + *
    ("based", "on", 1300),
    // before + *
    ("before", "the", 1500),
    // big + *
    ("big", "deal", 800),
    // by + *
    ("by", "the", 2200),
    // can + *
    ("can", "be", 2000),
    ("can", "do", 1400),
    // come + *
    ("come", "back", 1200),
    ("come", "here", 900),
    ("come", "on", 1100),
    ("come", "to", 1000),
    // could + *
    ("could", "be", 1800),
    ("could", "have", 1600),
    // did + *
    ("did", "you", 1400),
    // do + *
    ("do", "you", 1600),
    // don't + *
    ("don't", "know", 1700),
    // each + *
    ("each", "other", 1600),
    // even + *
    ("even", "though", 1500),
    // every + *
    ("every", "time", 900),
    // first + *
    ("first", "time", 1100),
    // for + *
    ("for", "the", 3000),
    // from + *
    ("from", "the", 2400),
    // get + *
    ("get", "out", 1000),
    ("get", "to", 1100),
    // go + *
    ("go", "back", 1100),
    ("go", "to", 1300),
    // going + *
    ("going", "to", 2400),
    // good + *
    ("good", "job", 900),
    ("good", "morning", 1000),
    ("good", "night", 950),
    // hard + *
    ("hard", "work", 800),
    // have + *
    ("have", "to", 2200),
    ("have", "you", 1400),
    // he + *
    ("he", "was", 1800),
    // hello + *
    ("hello", "world", 1200),
    // how + *
    ("how", "many", 1300),
    ("how", "much", 1400),
    // i + *
    ("i", "am", 2800),
    ("i", "can", 2200),
    ("i", "don't", 2600),
    ("i", "have", 2600),
    ("i", "think", 2400),
    ("i", "was", 2400),
    ("i", "will", 2200),
    // in + *
    ("in", "the", 4800),
    // into + *
    ("into", "the", 2000),
    // it + *
    ("it", "is", 2600),
    ("it", "was", 2400),
    // just + *
    ("just", "like", 1100),
    // kind + *
    ("kind", "of", 1700),
    // last + *
    ("last", "night", 1000),
    ("last", "year", 1100),
    // let + *
    ("let", "me", 1500),
    // long + *
    ("long", "time", 1200),
    // look + *
    ("look", "at", 1200),
    // looking + *
    ("looking", "for", 1300),
    // make + *
    ("make", "sure", 1300),
    // might + *
    ("might", "be", 1600),
    ("might", "have", 1400),
    // more + *
    ("more", "than", 1800),
    // most + *
    ("most", "of", 1600),
    // must + *
    ("must", "be", 1700),
    ("must", "have", 1500),
    // need + *
    ("need", "to", 2000),
    // new + *
    ("new", "york", 900),
    // next + *
    ("next", "time", 1000),
    ("next", "year", 1050),
    // no + *
    ("no", "one", 1500),
    // of + *
    ("of", "course", 1800),
    ("of", "the", 5000),
    // old + *
    ("old", "man", 850),
    // on + *
    ("on", "the", 2800),
    // one + *
    ("one", "of", 1900),
    // out + *
    ("out", "of", 1800),
    // over + *
    ("over", "the", 1800),
    // part + *
    ("part", "of", 1600),
    // put + *
    ("put", "on", 1000),
    // real + *
    ("real", "time", 900),
    // right + *
    ("right", "here", 1000),
    ("right", "now", 1900),
    // she + *
    ("she", "was", 1700),
    // should + *
    ("should", "be", 1700),
    ("should", "have", 1500),
    // so + *
    ("so", "far", 1400),
    ("so", "much", 1700),
    // some + *
    ("some", "of", 1600),
    // sort + *
    ("sort", "of", 1700),
    // such + *
    ("such", "as", 1600),
    // take + *
    ("take", "care", 1100),
    ("take", "on", 1000),
    // talk + *
    ("talk", "about", 1200),
    // tell + *
    ("tell", "me", 1400),
    // thank + *
    ("thank", "you", 2000),
    // that + *
    ("that", "is", 2200),
    // the + *
    ("the", "best", 1400),
    ("the", "end", 1300),
    ("the", "first", 1500),
    ("the", "last", 1400),
    ("the", "most", 1500),
    ("the", "next", 1300),
    ("the", "other", 1200),
    ("the", "people", 1600),
    ("the", "right", 1300),
    ("the", "same", 1600),
    ("the", "time", 1700),
    ("the", "way", 1600),
    ("the", "whole", 1200),
    ("the", "world", 1700),
    // there + *
    ("there", "are", 2000),
    ("there", "is", 2200),
    // they + *
    ("they", "are", 2000),
    ("they", "have", 1800),
    // think + *
    ("think", "about", 1200),
    // this + *
    ("this", "is", 2400),
    // through + *
    ("through", "the", 1800),
    // to + *
    ("to", "the", 3200),
    // too + *
    ("too", "much", 1500),
    // used + *
    ("used", "to", 1800),
    // want + *
    ("want", "to", 2200),
    // we + *
    ("we", "are", 2000),
    ("we", "have", 1900),
    // what + *
    ("what", "is", 2000),
    // who + *
    ("who", "is", 1700),
    // will + *
    ("will", "be", 2000),
    ("will", "do", 1400),
    // with + *
    ("with", "the", 2600),
    // work + *
    ("work", "on", 1100),
    // would + *
    ("would", "be", 2000),
    ("would", "have", 1800),
    ("would", "like", 1700),
    // you + *
    ("you", "are", 2400),
    ("you", "can", 2200),
    ("you", "have", 2000),
    ("you", "know", 2200),
    // young + *
    ("young", "man", 800),
];
