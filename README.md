# progettoASM

**Co-evoluzione di agenti LLM, rete sociale e fact-checking**

Progetto accademico per il corso di *Analisi di Social Network e Media*. Questo repository implementa una simulazione agent-based in cui agenti LLM leggono i post dei vicini, aggiornano il proprio stato, e influenzano una rete che viene a sua volta modificata da un link predictor GraphSAGE. Una fase separata seleziona e inietta fact-checker. Lo scenario tematico configurato è la diffusione di narrazioni polarizzanti in una rete di collaborazione accademica; non è una misura empirica di disinformazione reale.

Repository: [github.com/stefaano19/llm-social-network-simulation](https://github.com/stefaano19/llm-social-network-simulation)

**Contributo tecnico:** integrazione in un unico loop di agenti LLM, transizioni di stato di tipo Linear Threshold, perturbazione degli embedding, addestramento GraphSAGE/link prediction, rewiring a densità costante, checkpoint e metriche di struttura e opinione. Il notebook fornisce inoltre un confronto esplorativo tra una run di intervento e una di controllo. I risultati attualmente committati sono una sola realizzazione, senza repliche né inferenza statistica.

## Modello e loop

Il grafo iniziale deriva da `ogbl-collab` (Open Graph Benchmark), da cui il notebook estrae un sottografo; il valore configurato per l'esperimento descritto è 5.000 nodi con Forest Fire Sampling. La rete di co-autorialità viene usata come struttura di simulazione, non come rete di interazioni online o di diffusione osservata.

Gli agenti assumono uno dei quattro stati: **S** (suscettibile), **I** (diffonde la narrazione), **R** (resistente) e **F** (fact-checker). A ogni step:

1. Una frazione di agenti attivi legge il feed dei vicini e invia il contesto al backend LLM.
2. Lo stato proposto dall'LLM e la suscettibilità dichiarata concorrono alle regole di transizione Linear Threshold.
3. I cambi di stato perturbano gli embedding; GraphSAGE viene aggiornato sul grafo corrente e assegna score alle coppie di nodi.
4. Il rewirer sostituisce archi secondo gli score, mantenendo il numero di archi quando trova aggiunte e rimozioni compatibili; il nuovo grafo alimenta lo step successivo.

Nella Fase 3, CELF seleziona fino a `budget_k` nodi non infetti/non resistenti e li rende fact-checker. L'obiettivo predefinito `threshold` è un greedy deterministico del numero di nodi che superano le soglie immediate della state machine. L'alternativa `ic` usa Independent Cascade e simulazioni Monte Carlo.

## Implementazione

| Componente | Implementazione nel repository |
|---|---|
| Agenti | Python; client LLM per server OpenAI-compatible e Gemini, mock deterministico per dry-run |
| Backend documentato | vLLM locale; modello configurato `casperhansen/llama-3-8b-instruct-awq` |
| Grafo | NetworkX; caricamento OGB, estrazione Forest Fire/BFS/random walk/random, community detection |
| Link prediction | GraphSAGE a due layer con PyTorch; implementazione NumPy disponibile per dry-run senza training |
| Dinamica | Macchina a stati S/I/R/F e Linear Threshold modulato dalla suscettibilità LLM |
| Intervento | Greedy a soglia o CELF su Independent Cascade |
| Persistenza | CSV/JSON di risultati, log JSONL e checkpoint locali |

## Repository

```text
config.yaml                    Parametri centrali
requirements.txt               Dipendenze Python
notebooks/kaggle_full_run.ipynb Pipeline di esecuzione e confronto
src/orchestrator.py            Loop agenti → GNN → rewiring → metriche
src/agents/                    Agent, prompt, client LLM, seeder e state machine
src/graph/                     Dataset, campionamento, community e metriche
src/gnn/                       Embedding, GraphSAGE, training e rewiring
src/influence/                 Selezione, iniezione e metriche dei fact-checker
src/utils/                     Configurazione, seed, log e checkpoint
results/                       CSV/JSON di run effettivamente committati
relazione_accademica/          Relazione e presentazioni PDF
```

## Requisiti e installazione

La pipeline documentata è un notebook Kaggle configurato per vLLM e GPU. Richiede accesso a `ogbl-collab` tramite OGB, disponibilità del modello indicato su Hugging Face e un token `HF_TOKEN` come secret dell'ambiente. L'inferenza LLM richiede GPU e memoria adeguate al modello/quantizzazione; il notebook va adattato all'hardware disponibile.

```bash
git clone https://github.com/stefaano19/llm-social-network-simulation.git
cd llm-social-network-simulation
pip install -r requirements.txt
```

`requirements.txt` non installa PyTorch Geometric, che va installato separatamente secondo la versione di PyTorch/CUDA dell'ambiente (le note sono nel file). Il dataset, i pesi del modello, i checkpoint e le figure generate non sono inclusi. Il token va fornito tramite secret o variabile d'ambiente e non va committato. Per un dry-run senza inferenza reale impostare `USE_MOCK_LLM = True` nel notebook; il mock verifica il flusso software, ma non riproduce né valida i risultati con LLM reale.

## Configurazione ed esecuzione

I parametri della simulazione sono in `config.yaml`; i parametri di sessione sono nella prima sezione di `notebooks/kaggle_full_run.ipynb`.

| Parametro | Valore configurato/documentato | Significato |
|---|---:|---|
| `execution.random_seed` | `42` | Seed della simulazione |
| `subgraph.strategy`, `target_nodes`, `forward_prob` | `forest_fire`, `5000`, `0.5` | Campionamento rete |
| `simulation.activation_probability` | `0.15` | Quota di nodi attivati per step |
| `simulation.initial_infection_rate` | `0.15` | Quota iniziale di nodi I |
| `gnn.rewire_mode`, `max_new_edges_per_step` | `swap`, `50` | Rewiring a densità costante, fino a 50 scambi a evento |
| `influence.celf_objective`, `budget_k` | `threshold`, `20` | Obiettivo e budget dei fact-checker |
| `influence.fc_protection_threshold` | `0.10` | Soglia S → R dovuta alla pressione F |
| `influence.fc_resistance_threshold` | `0.25` | Soglia I → R dovuta alla pressione F |

Il notebook supporta run di Fase 2 in sessioni riprendibili da checkpoint; `PHASE2_STEPS` indica gli step da fare nella sessione corrente. Per confrontare intervento e controllo, l'intenzione del protocollo è partire dallo stesso checkpoint finale di Fase 2 e mantenere uguali `PHASE3_STEPS`, impostando `CONTROL_RUN` rispettivamente a `False` e `True`. La cella di confronto verifica i flag di controllo, il numero di step, lo step iniziale e i conteggi iniziali S/I/R ricostruiti dai delta; **non** verifica l'identità di grafo, embedding, pesi GNN, configurazione o stato del generatore casuale. Il checkpoint condiviso deve essere garantito dall'operatore.

I CSV sono append-only nelle run: usare directory/output distinti o ripulire manualmente solo gli artefatti di propria run, per evitare di mescolare sessioni non corrispondenti. Il notebook cerca report e CSV in `results/` e, su Kaggle, negli input montati per costruire `phase3_comparison.json`.

## Metriche

Le definizioni seguono `src/graph/metrics.py` e `src/influence/metrics.py`:

| Metrica | Definizione nel codice |
|---|---|
| Conteggi S/I/R/F, infection rate | Conteggio per stato; infection rate = nodi I / nodi totali |
| ECI | Media, sui nodi con vicini, della frazione di archi verso la stessa community |
| Modularità Q | Modularità del grafo rispetto alla mappa di community fornita |
| Opinion homophily | Quota di archi i cui estremi hanno lo stesso belief |
| Belief assortativity | Correlazione di Pearson dei belief agli estremi degli archi |
| Belief polarisation | Varianza dei belief normalizzata alla massima varianza teorica nell'intervallo configurato |
| Reach F | Quota di nodi non-F a distanza entro `reach_hops` da almeno un F |
| Copertura efficace F | Quota di nodi non-F adiacenti per cui la pressione F raggiunge le soglie della state machine |
| Transizioni | Numero di nodi che cambiano stato nel singolo step, come registrato nel CSV |

La codifica dei belief è S = 0, I = +1, R/F = −1. ECI e modularità usano la mappa di community iniziale; le metriche di opinione dipendono invece dagli stati correnti. Una crescita della polarizzazione indica una maggiore varianza dei belief, non necessariamente una migliore o peggiore accuratezza informativa.

## Risultati committati

Le misure numeriche sottostanti sono riportate nei CSV/JSON versionati in `results/`. Sono valori osservati in una singola simulazione, non benchmark, medie su repliche o stime con incertezza. `pipeline_summary_control.json` riporta seed 42, 5.000 nodi, 26.246 archi e 100 step di Fase 2; i report di Fase 3 descrivono gli step 100–129.

### Fase 2: step selezionati

| Step | S | I | R | ECI | Modularità Q | Homophily | Assortativity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3965 | 841 | 194 | 0.8324 | 0.8276 | 0.6290 | 0.2683 |
| 49 | 1070 | 2022 | 1908 | 0.8496 | 0.8538 | 0.7585 | 0.5408 |
| 99 | 978 | 2032 | 1990 | 0.8741 | 0.8769 | 0.8140 | 0.6586 |

Nel CSV gli archi restano 26.246; gli eventi di rewiring registrati scambiano 50 archi per evento (50 aggiunti e 50 rimossi). Questi dati mostrano l'andamento della run, non che il rewiring sia la causa delle variazioni osservate.

### Fase 3: intervento meno controllo al termine dello step 129

| Metrica | Intervento | Controllo | Differenza |
|---|---:|---:|---:|
| S | 798 | 974 | −176 |
| I | 2020 | 2024 | −4 |
| R | 2162 | 2002 | +160 |
| F | 20 | 0 | +20 |
| Infection rate | 0.4040 | 0.4048 | −0.0008 |
| ECI | 0.8855 | 0.8858 | −0.0003 |
| Modularità Q | 0.8831 | 0.8835 | −0.0004 |
| Homophily | 0.8305 | 0.8306 | −0.0001 |
| Assortativity | 0.7042 | 0.6922 | +0.0119 |
| Polarizzazione | 0.8394 | 0.8052 | +0.0342 |
| Transizioni totali nei 30 step | 177 | 17 | +160 |

Il report d'intervento indica CELF `threshold`, budget 20, 162 nodi coperti **stimati** in selezione e copertura efficace finale di 5 nodi (frazione 0.0010). La stima di selezione e la copertura misurata a fine run hanno definizioni e tempi diversi. Le 177/17 transizioni sono le somme della colonna `transitions` nei 30 step e non includono il semplice conteggio dei seed iniettati prima degli step.

Non si deve interpretare la tabella come effetto isolato del fact-checking: anche nel controllo proseguono LLM, GNN e rewiring. Il delta strutturale è quindi una differenza osservata tra due run dinamiche, non una misura dell'effetto causale dell'intervento. Sebbene il notebook controlli step e conteggi iniziali S/I/R, non certifica identità di tutti gli stati interni delle run.

### Artefatti dei risultati

| File | Contenuto |
|---|---|
| `results/metrics_history.csv` | Metriche per step della run di intervento |
| `results/metrics_history_control.csv` | Metriche per step della run di controllo |
| `results/phase3_report.json` | Report finale della run di intervento |
| `results/phase3_report_control.json` | Report finale della run di controllo |
| `results/phase3_comparison.json` | Differenze finali e conteggi/transizioni per step |
| `results/pipeline_summary_control.json` | Configurazione sintetica e riepilogo della run di controllo |

La relazione e le presentazioni sono in [`relazione_accademica/`](relazione_accademica/). I file generati di log JSONL, checkpoint e figure non sono inclusi; le regole in `.gitignore` li escludono lasciando tracciati i CSV/JSON dei risultati e i placeholder delle directory.

## Limitazioni

- **Una sola realizzazione per condizione:** mancano repliche con seed diversi, intervalli di incertezza e test statistici. Variazioni piccole non sono distinguibili dal rumore stocastico o dalla variabilità dell'LLM.
- **Confronto non pienamente controllato:** la cella del notebook verifica flag, durata, step iniziale e conteggi S/I/R ricostruiti, ma non hash di grafo, embedding, pesi GNN, configurazione o stato RNG.
- **Dinamica attiva nel controllo:** la run di controllo continua l'evoluzione LLM/GNN/rewiring; non costituisce un grafo statico né isola l'effetto del solo trattamento.
- **Greedy threshold:** l'obiettivo misura la copertura immediata rispetto a soglie locali. Non è submodulare, quindi non ha le garanzie di approssimazione standard di CELF; il codice esegue greedy completo.
- **Community statiche:** ECI e modularità sono calcolate usando la mappa iniziale di community.
- **Topologia:** gli swap conservano il numero di archi, non la connettività; la frammentazione è possibile.
- **Riproducibilità LLM:** i seed aiutano a riprodurre le componenti controllate dal codice, ma vLLM può non essere deterministico con batching variabile. Versioni non bloccate in `requirements.txt`, hardware e disponibilità del modello possono cambiare il comportamento.
- **Validità esterna:** rete di co-autorialità, narrazione simulata, prompt, stati discreti e agenti LLM sono astrazioni; i risultati non dimostrano dinamiche di utenti reali né efficacia reale del fact-checking.

## Riferimenti

- Dataset: [Open Graph Benchmark — `ogbl-collab`](https://ogb.stanford.edu/docs/linkprop/#ogbl-collab)
- Hamilton et al., *Inductive Representation Learning on Large Graphs* (GraphSAGE)
- Kempe, Kleinberg, Tardos, *Maximizing the Spread of Influence through a Social Network*
- Leskovec et al., *Cost-effective Outbreak Detection in Networks* (CELF)
- Serving LLM: [vLLM](https://github.com/vllm-project/vllm)
