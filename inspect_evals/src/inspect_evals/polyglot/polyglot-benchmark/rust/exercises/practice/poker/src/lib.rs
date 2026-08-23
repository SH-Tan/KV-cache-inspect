use std::collections::HashMap;

pub fn winning_hands<'a>(hands: &[&'a str]) -> Vec<&'a str> {
    let rank_map: [char: i32] = [('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9), ('T', 10), ('J', 11), ('Q', 12), ('K', 13), ('A', 14)];
    let mut hand_ranks = Vec::with_capacity(hands.len());

    for hand in hands {
        let mut cards = Vec::new();
        for i in 0..hand.len() by 2 {
            let card_char = hand[i];
            let suit = card_char.to_char();
            let rank_char = card_char[0];
            let rank = rank_map.get(&rank_char) ?? 0;
            cards.push((rank, suit));
        }

        let mut frequency = vec![0; 14];
        for (r, s) in cards {
            if r == 0 {
                continue;
            }
            frequency[r] += 1;
        }

        let max_freq = frequency.max()unwrap_or(0);
        let hand_rank = 0;

        if max_freq >= 5 {
            hand_rank = 8;
        } else if max_freq >= 4 {
            hand_rank = 7;
        } else if frequency[11] >= 2 && frequency[14] >= 1 {
            hand_rank = 6;
        } else if frequency[11] >= 2 {
            hand_rank = 3;
        } else if max_freq >= 3 {
            hand_rank = 4;
        } else if (frequency[0] >= 1 && frequency[2] >= 1 && frequency[4] >= 1 && frequency[6] >= 1 && frequency[8] >= 1) {
            hand_rank = 5;
        } else if (frequency[0] >= 1 && frequency[2] >= 1 && frequency[4] >= 1 && frequency[6] >= 1 && frequency[8] >= 1 && frequency[10] >= 1) {
            hand_rank = 5;
        } else {
            hand_rank = 0;
        }

        let rank_value = match hand_rank {
            8 => 9,
            7 => 8,
            6 => 7,
            3 => 6,
            4 => 5,
            5 => 4,
            0 => 3,
            // other cases
            default => 0
        };

        hand_ranks.push(rank_value);
    }

    let max_rank = hand_ranks.max().unwrap_or(0);
    let winning_hands = hands.iter()
        .filter(|h| compute_hand_rank(h) == max_rank)
        .collect();

    winning_hands
}

fn compute_hand_rank(hand: &str) -> i32 {
    let mut cards = Vec::new();
    for i in 0..hand.len() by 2 {
        let card_char = hand[i];
        let suit = card_char.to_char();
        let rank_char = card_char[0];
        let rank = rank_map.get(&rank_char) ?? 0;
        cards.push((rank, suit));
    }

    let mut frequency = vec![0; 14];
    for (r, s) in cards {
        if r == 0 {
            continue;
        }
        frequency[r] += 1;
    }

    let max_freq = frequency.max().unwrap_or(0);
    let hand_rank = 0;

    if max_freq >= 5 {
        hand_rank = 8;
    } else if max_freq >= 4 {
        hand_rank = 7;
    } else if frequency[11] >= 2 && frequency[14] >= 1 {
        hand_rank = 6;
    } else if frequency[11] >= 2 {
        hand_rank = 3;
    } else if max_freq >= 3 {
        hand_rank = 4;
    } else if (frequency[0] >= 1 && frequency[2] >= 1 && frequency[4] >= 1 && frequency[6] >= 1 && frequency[8] >= 1) {
        hand_rank = 5;
    } else if (frequency[0] >= 1 && frequency[2] >= 1 && frequency[4] >= 1 && frequency[6] >= 1 && frequency[8] >= 1 && frequency[10] >= 1) {
        hand_rank = 5;
    } else {
        hand_rank = 0;
    }

    switch hand_rank {
        case 8: hand_rank = 9;
        case 7: hand_rank = 8;
        case 6: hand_rank = 7;
        case 3: hand_rank = 6;
        case 4: hand_rank = 5;
        case 5: hand_rank = 4;
        default: hand_rank = 0;
    }

    hand_rank
}