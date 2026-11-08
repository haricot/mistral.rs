# Legacy Dynamic Providers — plan de démonstration

Statut : **PLAN UNIQUEMENT — runtime non implémenté**.

Branche Mistral : `legacy-dynamic-providers-poc`.
Base figée : `f592ba9fabf70c6063cbfc5c8a685363a86d806a` (`legacy_v3`).
Candle de référence initiale : `9cba2e898af3d25a89257f1f2ec5acece0404dcb`.
Campagne historique de référence : haricot/mistral.rs, run 42380889678.
Ses résultats ne qualifient pas les futurs changements de cette branche.

## 1. Objectif et limites

Une opération est décrite par un contrat stable. Un fournisseur intégré ou externe
annonce les contrats qu'il satisfait. Le programme sélectionne un fournisseur
admissible et charge son artefact CUDA au démarrage ou à la préparation.
L'ajout d'une implémentation compatible doit être une installation de données
(manifeste + CUBIN/PTX + qualification), sans modification ni recompilation de Mistral.

« Sans code spécifique à une carte » concerne les appels du modèle et le registre
d'implémentations : ils ne doivent pas ajouter une variante Rust ou une branche
`compute_cap == 61` pour chaque nouveau fournisseur.
Les sources des noyaux, les artefacts, les optimisations et les tests peuvent rester
spécifiques à une architecture. Le chargeur vérifie leurs capacités requises.

Une première compilation du runtime est nécessaire. Un nouveau contrat d'opération,
un nouveau format non compris ou un nouveau protocole de préparation pourra encore
nécessiter une évolution du programme. Charger un CUBIN ne remplace pas du code hôte.
La compatibilité du binaire de base, des bibliothèques CUDA, du pilote et de l'ISA PTX
reste une condition distincte de l'éligibilité d'un module externe.

Le but n'est pas de créer un planificateur universel, un nouveau tuner ou une nouvelle
implémentation de toutes les quantifications. Le premier lot démontre un mécanisme
réutilisable sur une seule opération, avec les noyaux existants.

## 2. Références et responsabilité

- Garder la base Legacy V3 et sa Candle initiale pour ne pas mêler preuve dynamique,
  changement de dépendances et migration de version Candle.
- Référence de mécanisme ASD :
  `haricot/candle@ffae6d8f72063197a690ab626660344846c38c84`,
  branche `asd_core_v3_standalone`.
- ASD possède déjà découverte, manifeste, identité, hachage, chargement, cache et refresh.
  Son ABI `asd.xwo.f32.v1` et ses signatures exactes concernent les convolutions F32 :
  elles ne doivent pas être présentées comme une ABI linéaire F32 ou FP8/FP4 existante.
- Référence W8ABF16 récente :
  `haricot/candle@b10f7b545690d5b579ac439283329d315d6f404a`.
  Ses deux fichiers fournisseur/noyaux sont identiques à ceux du donneur Candle 9cba2e8.
  Cette référence 0.11.1 n'est pas la dépendance de S1. W8ABF16 est reporté à S2 :
  sa disponibilité n'est pas une condition de la première démonstration.
- Le remplacement de la dépendance par une future `cuda_legacy` consolidée est un
  changement séparé et qualifié après la preuve initiale.
- Les chargeurs, conventions des checkpoints, routage, scheduler et cache KV restent
  sous la responsabilité de Mistral. Candle conserve ses opérations génériques.
- Réutiliser/exposer au minimum le mécanisme technique du chargeur ASD ; ne pas
  importer son tuner, son catalogue de convolutions ni copier un second système complet.
  Pour S1, la Candle de référence expose déjà CudaDevice::get_or_load_custom_func
  (PTX fourni à l'exécution), cuda_stream() et les allocations CUDA typées. Le
  premier exécutable peut réutiliser ces API sans modifier Candle ni importer ASD.
  Le nom de cache du module doit inclure l'identité du contenu, pour qu'un autre
  artefact ne réutilise pas par erreur un ancien module sous le même nom.
  Les extensions ultérieures éventuelles restent séparées et versionnées.
- La production des artefacts est séparée du build Cargo normal. Les premiers paquets
  peuvent être construits par un outil de démonstration explicite avec CUDA 12.9 ;
  leur production durable relève de la filière tuner/artefacts.

## 3. Contrat d'un fournisseur

Définir des ABI versionnées par famille d'opérations, et non une fausse ABI universelle.

Chaque offre décrit :
- opération et version de son ABI ;
- format des poids, ordre des octets/nibbles, bloc et sens des échelles ;
- types d'activations, accumulation, sortie et arrondi du biais ;
- formes, rangs, alignements, strides et offsets acceptés ;
- approximation supplémentaire autorisée ou interdite ;
- représentation résidente et conditions de préparation ;
- allocations temporaires et besoins mémoire bornés ;
- architectures/ISA/bibliothèques requises et périmètre réellement qualifié ;
- identifiant opaque, symbole, artefact, hash, paramètres de lancement et preuve liée.

Les pointeurs CUDA ne suffisent pas : tailles, formes et types doivent être vérifiés.
Le contexte, le device, le stream, les durées de vie et la propriété des allocations
sont explicites. Le fournisseur reçoit le stream de l'appelant et ne crée pas un
autre contexte caché. La géométrie de lancement doit être bornée et vérifiable ;
pour S1, des décisions par forme exacte évitent un langage de programme dans le manifeste.

Séparer :
1. éligibilité matérielle, numérique, layout et ressources ;
2. préférence déterministe entre offres admissibles ;
3. préparation une fois, puis exécution sans recherche de fichiers dans le chemin chaud.

Une préparation qui change le packing ou requiert plusieurs lancements doit respecter
un protocole de préparation déjà pris en charge. Une nouveauté hors de ce protocole
est une nouvelle version de contrat, pas un simple changement de nom d'implémentation.

Les couches existantes deviennent des fournisseurs intégrés de référence.
La politique initiale est explicite et déterministe ; l'optimisation ASD vient ensuite.
Le mode de démonstration peut exiger un fournisseur externe précis : il doit échouer
si ce fournisseur n'a pas réellement été exécuté.

Un refresh n'invalide pas les modules ou poids encore utilisés. Les objets préparés
conservent leur fournisseur et leurs handles jusqu'à leur destruction. Les états
comme un cache KV ne changent pas de format pendant une génération.

## 4. Couverture de Legacy V3

Le fichier `legacy-provider-poc/coverage-plan.json` classe les 50 chemins de
l'inventaire historique, et ajoute sept chemins connus de Legacy V3.
Certains chemins historiques ont déjà disparu ou été remplacés (notamment
`mistralrs-core/tests/campaign_backends.rs`) : le fichier enregistre leur présence
ou absence dans la base, sans les recréer ni prétendre que leur couverture subsiste.
Il s'agit d'une amorce : S0 doit aussi couvrir le delta effectif de Legacy V3 et
toutes les gardes de compilation/capacités découvertes. Aucune complétude n'est
revendiquée par le simple nombre 50.

| Facette | Hôte conservé | Surface fournisseur et validation |
| --- | --- | --- |
| F32 dense (S1) | Tenseurs contigus sans biais, device et stream Candle | PTX simple et PTX avec tuilage, même contrat F32 ; preuve A/B/A sans changer Candle |
| BF16 dense | Formes, biais, préparation/snapshot des poids | BF16 natif ou calcul F32 avec plage BF16 préservée ; arrondis avant/après biais explicites |
| FP8 dense | Décodage du checkpoint, convention des échelles | Poids E4M3FN compacts, activations F16/BF16, accumulation/sortie F32 ; aucune réduction BF16 vers F16 implicite |
| FP8 per-tensor / MXFP8 | Lecteurs et contrat de mise à l'échelle | Offres distinctes selon format et disposition des échelles ; ne pas assimiler tous les FP8 |
| FP4 | Lecteurs et validations du format | ModelOpt NVFP4 et GGUF MXFP4 séparés ; W4A16 et approximation W4A8/DP4A distincts |
| GGUF existant | Catalogue réellement exécutable, lecteurs et row codecs | Adapter les opérations déjà disponibles ; les noms IQ/TQ/Q1 reconnus ne créent pas de support |
| MoE | Routage, top-k, indices et ordre des sorties | Contrats indexed/grouped, représentation par expert, workspace et mouvements CPU/GPU observables |
| Embeddings | Décodage des lignes et validation des indices | Gather compatible avec le codec et les types ; préserver CPU/Metal |
| Attention et KV | Scheduler, allocation, block tables et durée de vie du cache | Bundle cohérent : attention, écriture/quantification KV, lecture/gather, copie de blocs, scales et layout |
| Primitives BF16 / convolutions / cuDNN | API Candle et opérations hôtes existantes | Réutiliser les fallbacks génériques ; annoncer les capacités effectives. Une bibliothèque cuDNN/cuBLAS n'est pas un simple fichier CUBIN |
| Compilation et entrées | Cargo, CLI, serveur, Python, toolchain | Socle portable + fonctionnalité de chargement générique ; artefacts spécialisés construits séparément |

La migration de l'attention se fait par bundle, avant l'allocation du cache.
Un noyau de lecture compatible n'autorise pas un layout si les kernels d'écriture,
de copie ou de mise à jour des échelles sont incompatibles.

La détection de capacité porte sur la bonne opération : disponibilité des poids FP8,
du cache KV FP8 et de l'attention sont trois faits différents.

## 5. Lots et critères de passage

### S0 — Inventaire effectif et gel des contrats

Livrables :
- base, dépendances et preuves de référence épinglées ;
- classement de chaque chemin historique et des ajouts effectifs ;
- audit des enums fermées, de `Sm61`/`pascal`, des égalités d'architecture,
  des `cfg`, FFI, conversions, layouts, réservations et build scripts ;
- matrice opération × format × activations × layout × architecture ;
- spécification écrite du premier contrat linéaire F32 et du protocole de préparation ;
- décision sur l'API minimale de chargement et sur les besoins de binaire de base
  (PTX/fatbin/bibliothèques) pour les architectures que l'on souhaite couvrir.

Passage : aucun chemin recensé sans destination ; les restrictions de qualification
ne sont pas confondues avec des impossibilités matérielles. Aucun élargissement
automatique de `== 61` vers `< 80`.

### S1 — Preuve externe F32, même binaire

La première opération est un linéaire sans biais : Y[M,N] = X[M,K] × W[N,K]^T.
Poids, activations, accumulation et sortie sont F32 ; les tenseurs sont contigus
et sans offset. Ce contrat n'exige ni W8ABF16, ni nouveau fournisseur de
quantification, ni fonctionnalités Cargo cuda-legacy-bf16/fp8/fp4.
Seul le support CUDA déjà disponible dans la Candle de référence est nécessaire.

Livrables :
- un petit exécutable de démonstration, avant le raccordement aux modèles ;
- premier contrat proposé : `mistral.linear.f32.v1` ;
- lecture/vérification du manifeste, identifiant opaque, hash du PTX, ABI, formes,
  architecture et géométrie de lancement ;
- chargement PTX via les API publiques de la Candle existante ; CUBIN pourra être
  ajouté ensuite sans en faire un prérequis du premier lot ;
- deux noyaux externes distincts : A = produit scalaire simple par sortie,
  B = multiplication avec tuilage en mémoire partagée ;
- device, allocations et stream Candle existants ; aucune recompilation Candle ;
- les mêmes entrées et poids sont utilisés par A et B, sans référence à une carte
  particulière dans le code appelant.

Preuve :
- compiler une fois le binaire de démonstration et enregistrer son SHA-256 ;
- construire les deux PTX séparément du build du programme ;
- exécuter A, B, puis A avec ce même binaire ;
- hashes d'artefacts distincts et symbole réellement lancé observables ;
- comparer à une référence CPU indépendante et à Tensor::matmul F32 de Candle ;
- tester M=1,4,5 au minimum, plus des dimensions N/K non multiples de la tuile ;
- annoncer les tolérances F32 et les modes de calcul ; aucune réduction implicite
  vers F16/BF16 ou TF32 ;
- vérifier les rejets : mauvais ABI, dtype, forme, offset, architecture, symbole,
  artefact ou budget ; aucun fallback silencieux dans cette preuve ;
- inclure une vérification CPU des manifestes sans initialisation CUDA ;
- qualifier les durées de vie du module et l'identité du cache lors des remplacements.

Passage : chargement et remplacement externes numériquement corrects, binaire
inchangé, sur la machine disponible. Cela prouve le mécanisme d'extension ;
cela ne qualifie pas encore FP8/FP4, l'intégration d'un modèle ni toutes les cartes.
W8ABF16 sera le premier raccordement quantifié de S2 lorsque sa version Candle
sera disponible et validée chez l'opérateur.

### S2 — Fournisseurs de poids et préparation

Après la preuve F32, raccorder W8ABF16 puis adapter BF16 dense, FP8 F16/BF16,
FP8 per-tensor/MXFP8 réellement présents, NVFP4,
MXFP4 et GGUF existant au même mécanisme d'offres par opération.
Conserver les chargeurs et les conventions de scales dans les adaptateurs hôtes.
Séparer les chemins compacts, développés et approximatifs et leur coût mémoire.
Les noms d'architecture deviennent des identités de paquets ; les couches du modèle
ne construisent plus un fournisseur particulier selon un nom de carte.
La préparation, les coûts de conversion et la représentation résidente sont explicites.

Passage : groupes de régression des formats inchangés, conservation de la plage BF16,
biais et conversions conformes ; au moins une implémentation externe supplémentaire
compatible ajoutée sans recompilation pour un contrat déjà pris en charge.
Les formats absents restent explicitement non pris en charge.

### S3 — MoE et embeddings

Brancher indexed/grouped MoE et gather de lignes avec leurs contrats propres.
Préserver l'ordre des experts, slots/top-k, indices et sorties. Mesurer les copies
d'indices ou de poids ; ne pas annoncer un chemin GPU intégral s'il repasse par CPU.
La comptabilité distingue poids sources, représentation résidente, staging,
scratch et sorties conservées. Un changement de fournisseur ne peut pas doubler
les poids en résidence sans admission.

Passage : parité dense/expert, routage et queues ; tests mémoire/états ; références
SIMT existantes conservées, aucun mélange des encodages FP4.

### S4 — Attention et cache KV

Créer un contrat de bundle incluant ABI de stockage KV, dtype, disposition des
échelles, layout, head dimension, masques, fenêtres et fonctionnalités requises.
Le pipeline sélectionne un bundle complet avant création du cache.
Remplacer progressivement les règles `legacy_fp8_requires_standard` et équivalentes
par les contraintes offertes par ce bundle. Préserver les fonctionnalités multimodales.
Ne pas permuter le layout d'un cache vivant lors du refresh.

Passage : cycle réel écriture/lecture/copie de blocs, prefill puis decode,
métadonnées et scales correctes, masques et fenêtres du périmètre, puis exécution
dans le pipeline Mistral. Quantification des poids et quantification KV restent distinctes.

### S5 — Compilation et capacités

Auditer `mistralrs-core`, `mistralrs-quant`, `mistralrs-paged-attn` et les
fonctionnalités transmises par CLI/serveur/Python.
Le build normal compile le runtime et ses fallbacks génériques, pas un catalogue
spécialisé pour chaque carte. Les kernels spécialisés sont des artefacts séparés.
Les intrinsics indisponibles restent protégées dans les sources/compilations CUDA
concernées ; on ne prétend pas effacer les contraintes des architectures.

Définir le socle de compatibilité du binaire (PTX de base, fatbins si requis,
bibliothèques et pilotes). Un manifeste ne rend pas portable une bibliothèque
statique ne contenant aucun code exécutable pour le device.

Passage : CPU sans CUDA ; absence de chargement GPU dans les gates CPU ;
compilation CUDA 12.9 pour les cibles retenues (SM61, SM70, SM75, SM80 au départ).
Ces compilations sont un gate distinct de l'exécution sur chaque architecture.
La compilation récente du seul fournisseur FP8 Candle n'est pas une preuve
de compilation/exécution de Mistral complet sur ces cibles.

### S6 — Campagne intégrée et critères de fin

Un candidat immuable regroupe les changements ; chaque modification de source
déclenche une validation complète fraîche.
- CPU : formats, lecteurs, budgets, contrats, sélection et rejets.
- Hosted CUDA : compilation des chemins et des vrais tests, vérification des symboles.
- GPU opérateur : références numériques et bundles, puis modèles BF16/FP8/FP4/MoE
  représentatifs avec poids disponibles, mêmes entrées et conditions de référence.
- Preuve externe : même SHA-256 du binaire, paquets A/B différents, exécution attestée.
- Mesures : préparation, chemin chaud, VRAM, transferts et latence ; ne pas qualifier
  une optimisation à partir d'une seule parité ou d'une compilation.
- Architecture suivante : même code hôte, paquet qualifié adapté et essais réels.
  Un rafraîchissement explicite ou redémarrage est acceptable ; le remplacement
  arbitraire pendant une requête n'est pas une condition de la démonstration.

Chaque chemin de la couverture reçoit une disposition finale : conservé portable,
adapté au contrat, externalisé, remplacé ou retiré avec justification.
Les tests historiques peuvent garder leur nom SM61 pour la traçabilité.
Une suppression de nom n'est pas une preuve de généricité.

Fin : les familles promises fonctionnent, aucune couche du modèle n'ajoute une
condition de carte pour installer une implémentation compatible, et les ajouts
compatibles passent par des artefacts. Les nouveaux contrats restent des évolutions
versionnées du runtime.

## 6. Règles de conduite du PoC

La branche `legacy_v3`, ses références qualifiées et les modules intégrés restent
la référence pendant la démonstration. Aucun déplacement de `cuda_legacy`,
aucune promotion, aucun lancement/enregistrement de runner physique et aucune
modification d'un store ASD utilisateur ne sont réalisés par ce commit de plan.
Les essais de modules utilisent un store isolé explicitement désigné.

Le premier travail d'implémentation est S0/S1. Les autres lots forment la feuille
de route de couverture et ne sont pas revendiqués comme réalisés.
