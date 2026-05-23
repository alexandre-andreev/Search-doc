(venv-embed-test) PS D:\_project\Search-doc> python embedding_benchmark.py

======================================================================
  „€ЂѓЌЋ‘’€ЉЂ EMBEDDING-‘’…ЉЂ
======================================================================
  –Ґ«Ґў®Ґ ¦Ґ«Ґ§®: GTX 1650 Ti (4 GB VRAM)
  –Ґ«Ґў п § ¤ з : Ё­¤ҐЄб жЁп ~3500 Є­ЁЈ ? 1.75M з ­Є®ў
  ђ §¬Ґа вҐбв®ў®Ј® ­ Ў®а : 500 з ­Є®ў

--- ‘ЁбвҐ¬  ---
  platform: Windows-11-10.0.26200-SP0
  python: 3.13.5
  ram_total_gb: 15.8
  ram_available_gb: 4.9

--- PyTorch / CUDA ---
    torch_version: 2.6.0+cu124
  ? cuda_available: True
    cuda_version: 12.4
    gpu_name: NVIDIA GeForce GTX 1650 Ti
    vram_total_gb: 4.0
    compute_capability: 7.5

--- sentence-transformers ---
  ? ўҐабЁп: 5.5.1

======================================================================
  GPU Ѓ…Ќ—ЊЂђЉ
======================================================================

--- Њ®¤Ґ«м: intfloat/multilingual-e5-small  (118M Ї а ¬Ґва®ў) ---
  Ћ¦Ё¤ Ґ¬л© VRAM ў FP32: ~0.5 GB
  ‡ ¬ҐвЄ : ‘ ¬ п Ўлбва п, ¤«п baseline. Љ зҐбвў® ­Ё¦Ґ, ­® ­  1650Ti аҐ «ЁбвЁз­ .
  ‡ Јаг¦ Ґ¬ ¬®¤Ґ«м ­  cuda... Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
modules.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 387/387 [00:00<00:00, 2.26MB/s]
D:\_project\Search-doc\venv-embed-test\Lib\site-packages\huggingface_hub\file_download.py:138: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\alexa\.cache\huggingface\hub\models--intfloat--multilingual-e5-small. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
README.md: 498kB [00:00, 7.11MB/s]
sentence_bert_config.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 57.0/57.0 [00:00<00:00, 393kB/s]
config.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 655/655 [00:00<00:00, 3.59MB/s]
model.safetensors: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 471M/471M [00:40<00:00, 11.7MB/s]
Loading weights: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 199/199 [00:00<00:00, 7531.12it/s]
tokenizer_config.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 443/443 [00:00<00:00, 2.45MB/s]
sentencepiece.bpe.model: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 5.07M/5.07M [00:02<00:00, 2.36MB/s]
tokenizer.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 17.1M/17.1M [00:01<00:00, 9.08MB/s]
special_tokens_map.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 167/167 [00:00<00:00, 1.02MB/s]
config.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 200/200 [00:00<00:00, 1.39MB/s]
Ј®в®ў® §  55.1 бҐЄ
  Џа®ЈаҐў (16 з ­Є®ў)... ®Є
  batch_size=  8:  356.1 з ­Є®ў/бҐЄ  (1.4 бҐЄ ­  500 з ­Є®ў, peak VRAM 468 MB)
  batch_size= 16:  383.9 з ­Є®ў/бҐЄ  (1.3 бҐЄ ­  500 з ­Є®ў, peak VRAM 479 MB)
  batch_size= 32:  373.2 з ­Є®ў/бҐЄ  (1.3 бҐЄ ­  500 з ­Є®ў, peak VRAM 501 MB)

--- Њ®¤Ґ«м: intfloat/multilingual-e5-base  (280M Ї а ¬Ґва®ў) ---
  Ћ¦Ё¤ Ґ¬л© VRAM ў FP32: ~1.1 GB
  ‡ ¬ҐвЄ : •®а®иЁ© Є®¬Їа®¬Ёбб бЄ®а®бвм/Є зҐбвў®. ‘Є®аҐҐ ўбҐЈ® ®ЇвЁ¬ «м­л© ўлЎ®а.
modules.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 387/387 [00:00<00:00, 1.96MB/s]
D:\_project\Search-doc\venv-embed-test\Lib\site-packages\huggingface_hub\file_download.py:138: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\alexa\.cache\huggingface\hub\models--intfloat--multilingual-e5-base. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
README.md: 179kB [00:00, 7.62MB/s]
sentence_bert_config.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 57.0/57.0 [00:00<00:00, 309kB/s]
config.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 694/694 [00:00<00:00, 2.87MB/s]
model.safetensors: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 1.11G/1.11G [01:26<00:00, 12.9MB/s]
Loading weights: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 199/199 [00:00<00:00, 7789.85it/s]
tokenizer_config.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 418/418 [00:00<00:00, 2.67MB/s]
sentencepiece.bpe.model: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 5.07M/5.07M [00:02<00:00, 2.50MB/s]
tokenizer.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 17.1M/17.1M [00:01<00:00, 15.3MB/s]
special_tokens_map.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 280/280 [00:00<00:00, 1.40MB/s]
config.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 200/200 [00:00<00:00, 890kB/s]
Ј®в®ў® §  99.3 бҐЄ
  Џа®ЈаҐў (16 з ­Є®ў)... ®Є
  batch_size=  8:  113.4 з ­Є®ў/бҐЄ  (4.4 бҐЄ ­  500 з ­Є®ў, peak VRAM 1092 MB)
  batch_size= 16:  107.7 з ­Є®ў/бҐЄ  (4.6 бҐЄ ­  500 з ­Є®ў, peak VRAM 1113 MB)
  batch_size= 32:  113.2 з ­Є®ў/бҐЄ  (4.4 бҐЄ ­  500 з ­Є®ў, peak VRAM 1156 MB)

--- Њ®¤Ґ«м: intfloat/multilingual-e5-large  (560M Ї а ¬Ґва®ў) ---
  Ћ¦Ё¤ Ґ¬л© VRAM ў FP32: ~2.2 GB
  ‡ ¬ҐвЄ : ‹гзиҐҐ Є зҐбвў®. Ќ  4GB VRAM ўЇаЁвлЄ - ¬®¦Ґв Ї®ваҐЎ®ў вмбп ¬ «Ґ­мЄЁ© Ў вз.
modules.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 387/387 [00:00<00:00, 2.63MB/s]
D:\_project\Search-doc\venv-embed-test\Lib\site-packages\huggingface_hub\file_download.py:138: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\alexa\.cache\huggingface\hub\models--intfloat--multilingual-e5-large. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
README.md: 160kB [00:00, 25.5MB/s]
sentence_bert_config.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 57.0/57.0 [00:00<00:00, 384kB/s]
config.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 690/690 [00:00<00:00, 4.36MB/s]
model.safetensors: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 2.24G/2.24G [02:49<00:00, 13.2MB/s]
Loading weights: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 391/391 [00:00<00:00, 5684.21it/s]
tokenizer_config.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 418/418 [00:00<00:00, 2.97MB/s]
sentencepiece.bpe.model: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 5.07M/5.07M [00:02<00:00, 2.17MB/s]
tokenizer.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 17.1M/17.1M [00:01<00:00, 8.97MB/s]
special_tokens_map.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 280/280 [00:00<00:00, 2.02MB/s]
config.json: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 201/201 [00:00<00:00, 1.18MB/s]
Ј®в®ў® §  185.1 бҐЄ
  Џа®ЈаҐў (16 з ­Є®ў)... ®Є
  batch_size=  8:   37.6 з ­Є®ў/бҐЄ  (13.3 бҐЄ ­  500 з ­Є®ў, peak VRAM 2175 MB)
  batch_size= 16:   32.9 з ­Є®ў/бҐЄ  (15.2 бҐЄ ­  500 з ­Є®ў, peak VRAM 2204 MB)
  batch_size= 32:   32.6 з ­Є®ў/бҐЄ  (15.3 бҐЄ ­  500 з ­Є®ў, peak VRAM 2262 MB)

======================================================================
  CPU Ѓ…Ќ—ЊЂђЉ (Є®­ва®«м­л©, ¤«п ба ў­Ґ­Ёп)
======================================================================

  (­  CPU Ј®­Ё¬ в®«мЄ® 100 з ­Є®ў ¤«п нЄ®­®¬ЁЁ ўаҐ¬Ґ­Ё)

--- Њ®¤Ґ«м: intfloat/multilingual-e5-small  (118M Ї а ¬Ґва®ў) ---
  Ћ¦Ё¤ Ґ¬л© VRAM ў FP32: ~0.5 GB
  ‡ ¬ҐвЄ : ‘ ¬ п Ўлбва п, ¤«п baseline. Љ зҐбвў® ­Ё¦Ґ, ­® ­  1650Ti аҐ «ЁбвЁз­ .
Loading weights: 100%|ЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ| 199/199 [00:00<00:00, 4811.81it/s]
Ј®в®ў® §  7.4 бҐЄ
  Џа®ЈаҐў (16 з ­Є®ў)... ®Є
  batch_size=  8:   44.3 з ­Є®ў/бҐЄ  (2.3 бҐЄ ­  100 з ­Є®ў)

======================================================================
  ђ…ЉЋЊ…Ќ„Ђ–€€
======================================================================

  ‹гзиЁ© аҐ§г«мв в ¤«п Є ¦¤®© ¬®¤Ґ«Ё ­  GPU:

  Њ®¤Ґ«м        Ѓ вз   — ­Є®ў/бҐЄ   VRAM peak   €­¤ҐЄб жЁп 1.75M з ­Є®ў
  ------------ ----- ------------ -----------   -------------------------
  e5-small        16        383.9      479 MB   1.27 з
  e5-base          8        113.4     1092 MB   4.29 з
  e5-large         8         37.6     2175 MB   12.93 з

  ? „«п ЎЁЎ«Ё®вҐЄЁ 3500 Є­ЁЈ аҐЄ®¬Ґ­¤гҐвбп e5-base б batch_size=8.
    ќв® а §г¬­л© Є®¬Їа®¬Ёбб ¬Ґ¦¤г Є зҐбвў®¬ Ё бЄ®а®бвмо ­  GTX 1650 Ti.
    e5-large в®¦Ґ а Ў®в Ґв, ­® ў 3.0x ¬Ґ¤«Ґ­­ҐҐ. ЃҐаЁвҐ Ґс, в®«мЄ® Ґб«Ё ўл¤ з  e5-base ®Є ¦Ґвбп б« Ў®©.

======================================================================
  ѓ®в®ў®. ‘®еа ­ЁвҐ ўлў®¤ нв®Ј® бЄаЁЇв  - ®­ Ї®­ ¤®ЎЁвбп ¤«п ¤Ё§ ©­  Ё­¤ҐЄб .
======================================================================
