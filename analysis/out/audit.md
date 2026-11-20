# Split audit

Columns: `image_id, genre, emotion, caption, art_style, painting, language, image_name, split`

## Flags

- None.

## Splits

| metric | train | val |
|---|---|---|
| paintings | 1572 | 86 |
| annotations | 1.65e+05 | 8732 |
| st1_instances | 1.524e+05 | 8057 |
| languages_per_painting_median | 26 | 26 |
| st3_pairs | 1.602e+04 | 849 |
| st3_paintings_with_3plus_languages | 1532 | 83 |
| year_coverage | 0.5274 | 0.593 |


## Era counts (paintings)

| era | train | val |
|---|---|---|
| unknown | 743 | 35 |
| 1870-1909 | 298 | 14 |
| 1910-1944 | 191 | 7 |
| 1945+ | 152 | 13 |
| <1600 | 73 | 5 |
| 1800-1869 | 61 | 7 |
| 1600-1799 | 54 | 5 |


## Artists

{'train': 521, 'val': 75, 'val_paintings_with_train_artist': 69}


## Is val unusual? (TVD vs. random painting draws)

| attribute | tvd | null_median | p_value | statistic |
|---|---|---|---|---|
| art_style | 0.2153 | 0.1900 | 0.2198 |  |
| genre | 0.1868 | 0.1158 | 0.0350 |  |
| era | 0.1326 | 0.0921 | 0.1079 |  |
| artist | 0.7406 | 0.7287 | 0.3207 |  |
| year_ks |  |  | 0.4902 | 0.1168 |


## Phase inputs (IDs only)

```
{'dev': {'st1': {'rows': 16114, 'columns': ['id', 'language', 'image', 'caption'], 'languages': ['Arabic', 'Burmese', 'Chinese', 'Darija', 'Emakhuwa', 'English', 'Hausa', 'Hindi', 'Igbo', 'Indonesian', 'IsiNdebele', 'IsiXhosa', 'IsiZulu', 'Kazakh', 'Kinyarwanda', 'Korean', 'Kyrgyz', 'Malay', 'Setswana', 'Swahili', 'Tagalog', 'Tamil', 'Thai', 'Turkish', 'Urdu', 'Uzbek', 'Vietnamese', 'Yoruba']}, 'st2': {'rows': 2320, 'columns': ['painting', 'source_language', 'target_language', 'source_p_positive'], 'paintings': 83, 'paintings_in_train': 0, 'paintings_with_train_artist': 66, 'year_coverage': 0.6024096385542169, 'era_counts': {'unknown': 33, '1870-1909': 14, '1945+': 12, '1800-1869': 7, '1910-1944': 7, '1600-1799': 5, '<1600': 5}}, 'st3': {'rows': 849, 'columns': ['painting', 'language', 'image'], 'paintings': 85, 'paintings_in_train': 0, 'paintings_with_train_artist': 68, 'year_coverage': 0.6, 'era_counts': {'unknown': 34, '1870-1909': 14, '1945+': 13, '1800-1869': 7, '1910-1944': 7, '1600-1799': 5, '<1600': 5}, 'languages': ['Arabic', 'Burmese', 'Chinese', 'English', 'Hausa', 'Hindi', 'Igbo', 'Indonesian', 'IsiZulu', 'Korean', 'Kyrgyz', 'Malay', 'Setswana', 'Swahili', 'Tagalog', 'Tamil', 'Thai', 'Turkish', 'Urdu', 'Uzbek', 'Vietnamese', 'Yoruba']}}, 'test': {}}
```

Dev ST3 file vs. val gold pairs: {'dev_pairs': 849, 'gold_pairs': 849, 'in_both': 849}


Test-phase unscored languages: ['Arabic', 'Chinese', 'English']
