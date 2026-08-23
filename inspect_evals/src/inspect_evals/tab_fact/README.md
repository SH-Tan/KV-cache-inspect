# tab_fact

TABFACT is large scale dataset with 16k Wikipedia tables as evidence for 118k human annotated statements designed for fact verification with semi-structured evidence. The statements are labeled as either ENTAILED or REFUTED. TABFACT is challenging since it involves both soft linguistic reasoning and hard symbolic reasoning.

## Dataset

Example sample:

```
{'id': 0,
 'table': {'id': '2-1570274-4.html.csv',
  'header': ['tournament',
   'wins',
   'top - 5',
   'top - 10',
   'top - 25',
   'events',
   'cuts made'],
  'rows': [['masters tournament', '0', '1', '2', '4', '4', '4'],
   ['us open', '0', '2', '3', '4', '6', '5'],
   ['the open championship', '1', '2', '2', '2', '3', '3'],
   ['pga championship', '0', '0', '1', '2', '5', '4'],
   ['totals', '1', '5', '8', '12', '18', '16']],
  'caption': 'tony lema'},
 'statement': 'tournament that tony lema have not participate in include the master tournament , the us open , the pga championship and the open championship',
 'label': 0}
 ```

The prompt is mostly taken from [here](https://huggingface.co/datasets/madrylab/platinum-bench/viewer/tab_fact).
