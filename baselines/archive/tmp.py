from IPython import embed
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '/share/shared_models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33'

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

text = tokenizer.decode([    1, 29871,    13, 29961, 25580, 29962,   673,   278,  1494,  1139,
        29889,  8797,   368,  1962,   278,  1234,  1728,   738,   916,  3838,
        29889,    13,    13, 16492, 29901,  1724,   947,   278,   360, 17061,
        29899,  5667,  4529,   408, 29901, 29871, 17891, 29909, 29889,   319,
          639,  4428,   573,  1320, 23575,   322,  8872, 14803,  2264,   310,
         4045,  1316,   393,  1009,  3184,  3145,   526, 21551,   408, 14263,
         1555,   296, 29892,  6763,   491,  4688,   594,   352,   386,  2092,
          322,  2198,   297,   263, 12875,   310,  3030, 29879, 29892,   408,
        18694,   491,  3023,   313,   272,   901, 29897,   310,   278,  1494,
        29901,   313, 29896, 29897, 12326, 29879, 29892,  1728,  8002,  8405,
        29892,   393,  4045,   526, 16035, 11407, 29892, 10311,   292, 29892,
          470, 23332,  4357,  1075,   313, 29906, 29897,   338,   758, 16770,
         1000,   411,   443,  5143,  2164, 27455,  1372,  1048,   278, 28108,
         1017,   470,  9311, 12554,  3335,   310,  7875,   470,  4067,  1078,
          313, 29941, 29897,   338,  1104,  5313,   424,   304,  1970,   680,
          297,  4045,  1363,   310, 18500,  2749,  9714,  8866,   393,   278,
         2472,   674,   367,  1304,  4439,  1654,  5794,  2750,  1075,   313,
        29946, 29897, 13623,  7934,   316, 12676,   292,   470, 20616,   292,
         2099,   886,   964,  3856,   647, 29360,   470,  4959,   313, 29945,
        29897, 24379,  2705,   367,  1503, 21227,  2710, 29892,   474, 29889,
        29872,  1696,   338, 29395,   990,  4357,   310,  1663,   499, 29879,
        29892, 10899, 14886, 29892,   470,  2243,  5861,   313, 29953, 29897,
        17189,  3145, 16661,   373,   670,  2931,   470, 19821,   393,   526,
          451, 20295,   304,  4045,   322,   338,  4996,   304,  7657,   385,
          629,  2354,   470,   304,  6795,  1131,   547,   313, 29955, 29897,
          756,  1162,  1264,  8872,   293,  1080])
embed()
input()