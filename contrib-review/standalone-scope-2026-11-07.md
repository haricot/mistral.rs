# Revue des standalones pour des contributions sans socle Legacy

Photographie du 7 novembre 2026. Revue statique de `haricot/mistral.rs`.
Master du fork : `095991a7d680a6994e6f1554156ebef5d09db049`. Donneur Legacy : `e4f7349cf973b5603ccbfe344e90c4d019a62754`.

## Avis et regle de separation

Oui : documenter les retraits avant de creer les versions contrib evite de
reproduire le socle Legacy dans chaque fonctionnalite. Garder une note de revue
avec des SHA et des chemins precis est preferable a des TODO disperses dans le
code. Les commentaires proposes ci-dessous concernent une future extraction ;
aucun bloc n'est supprime par cette revue.

Une version contrib contient la fonctionnalite et ses prerequis minimaux
explicites. Le support Pascal commun appartient au support de compatibilite ou
a une contribution distincte. Un algorithme portable, un garde de capacite utile
ou un format de checkpoint ne deviennent pas inutiles parce qu'ils servent
aussi sur Pascal. Les standalones BF16/FP8/FP4 v2 ont justement une fonction
Pascal : leur coeur ne doit pas etre supprime au titre d'un nettoyage Legacy.

Categories de revue : **CONSERVER**, **RETIRER DU PATCH CONTRIB**,
**PREREQUIS A ISOLER**, **REQUALIFIER**. Retirer un delta ne signifie jamais
supprimer le fichier amont entier, notamment pour Cargo, les build.rs et les
modules partages.

## Inventaire des sept autres standalones

| Branche | SHA examine | Fichiers modifies depuis la base commune | Legacy dans l'ascendance | Fichiers modifies identiques a Legacy |
| --- | --- | ---: | --- | ---: |
| `asd_runner_standalone` | `e3ce54f7a7f65b6046407fb206d294f5070dad21` | 59 | oui | 45 |
| `lfm25_dspark_standalone` | `c0e7282e2bfc6e50979811749c8e24d11fcb5a9a` | 73 | oui | 44 |
| `nemotron_bolt_dflash_standalone` | `9ff03fa63e624cf919108480df9b6ff3b1f4f4b2` | 71 | oui | 47 |
| `UQFF_draft_standalone` | `53aefbde829a0e8576e3e6b89c9f4da814772bf0` | 57 | non | 11 |
| `embeddinggemma2_standalone` | `641dd2541406476203c286c8d4cd5e431951c103` | 11 | non | 0 |
| `cpu-moe-standalone` | `9e39c6c219bc390038e532ddcacec6043d5b8957` | 11 | non | 0 |
| `uqff_shard_stream_upstream_standalone` | `c00fc25f6022cddf2f179388b820c441cc36d7a4` | 1 | non | 0 |

Les correspondances exactes portent sur les chemins modifies par Legacy contre
master, compares aux blobs des standalones. Elles sont un indice d'import, pas
une liste automatique de suppressions. Les anciens points de depart de Draft,
EmbeddingGemma 2 et UQFF streaming sont conserves dans le fichier JSON joint.

## Commentaires proposes par branche

### ASD

**RETIRER DU PATCH CONTRIB** : le merge transitif Legacy, ses features
cuda-legacy-bf16/fp8/fp4, le cache KV FP8 Pascal, les GEMM MoE SIMT et les
adaptateurs de quantification sans rapport avec la contribution ASD. Conserver
ces apports dans leurs proprietaires respectifs. Les modifications globales de
CI, les epingles de campagne et les corrections RoPE doivent etre examinees
separement ; elles ne definissent pas une integration ASD.

**PREREQUIS A ISOLER** : le backend ASD appartient a Candle. Une contribution
Mistral minimale doit declarer le backend necessaire et prouver le chemin
operateur effectivement consomme. Ne pas remplacer le fork par crates.io avant
d'avoir identifie et fourni l'API/backend ASD requis.

**REQUALIFIER** : le test actuel [asd_sm61_smoke](https://github.com/haricot/mistral.rs/blob/e3ce54f7a7f65b6046407fb206d294f5070dad21/mistralrs-core/tests/asd_sm61_smoke.rs#L6)
calcule `data + data` sur CUDA. Il ne prouve ni selection ASD, ni chargement
d'un artefact ASD, ni convolution groupee, ni gain de performance.
De plus, [le preflight ASD](https://github.com/haricot/mistral.rs/blob/e3ce54f7a7f65b6046407fb206d294f5070dad21/.github/workflows/asd-standalone-preflight.yml#L42)
et le workflow de resolution du lock exigent encore Candle `66c4651...`, tandis
que [le manifeste actuel](https://github.com/haricot/mistral.rs/blob/e3ce54f7a7f65b6046407fb206d294f5070dad21/mistral-integration/sources.json#L5)
et Cargo.toml epinglent `e0dbd926...`. Ce controle de provenance est incoherent
avant meme la compilation. Un futur controle doit utiliser l'entree convenue
pour la contribution et le lockfile qui lui correspond.

> Commentaire de revue : reconstruire une contribution ASD minimale depuis le
> master cible, garder le backend Candle ASD comme prerequis explicite, et
> exclure les deltas Legacy et les ajustements de campagne. Remplacer la preuve
> d'addition CUDA par une preuve du vrai dispatch ASD avant de qualifier ASD.

### LFM2.5 / DSpark

**CONSERVER** : [le runtime DSpark natif](https://github.com/haricot/mistral.rs/blob/c0e7282e2bfc6e50979811749c8e24d11fcb5a9a/mistralrs-core/src/models/lfm2/speculative.rs#L74),
la reservation des checkpoints recurrents, ShortConv et le rollback, la
geometrie du cache et des requetes multi-tokens, la verification autoregressive
et le scheduler de confiance. Garder les modifications d'inputs_processor,
paged_attention/layers et speculative qui assurent ces contrats : leur emplacement
partage n'en fait pas un residu Legacy.

**RETIRER DU PATCH CONTRIB** : le socle Legacy repris en transit, ses changements
de build CUDA et de formats non requis par DSpark, les hooks KV FP8 SM61 et les
features Legacy propagees dans tous les crates. Retirer la dependance de
[prerequisites.legacy](https://github.com/haricot/mistral.rs/blob/c0e7282e2bfc6e50979811749c8e24d11fcb5a9a/mistral-integration/sources.json#L15)
dans le nouveau contrat, uniquement apres extraction du vrai delta DSpark.
La [recette CUDA de campagne](https://github.com/haricot/mistral.rs/blob/c0e7282e2bfc6e50979811749c8e24d11fcb5a9a/.github/workflows/lfm25-dspark.yml#L66)
compile explicitement les trois features Legacy pour SM61 ; ce n'est pas le
gate autonome d'une contribution generique.

**REQUALIFIER** : README et manifeste ne decrivent plus les memes bases/Candle.
Le README cite `20f607a...` et `f5ce7f6...`, le manifeste cite master `095991a...`
et Candle `e0dbd926...`. La contribution doit prendre ses references dans les
sources exactes retenues, sans recycler les anciennes preuves.

> Commentaire de revue : conserver DSpark et ses contrats de cache/ShortConv,
> extraire le socle de compatibilite Pascal, puis construire et tester DSpark
> contre le master et la dependance Candle explicitement retenus. Les mesures
> de confiance et de latence restent liees au modele et au materiel observes.

### Nemotron Bolt / DFlash

**CONSERVER** : architecture GGUF Bolt/Gemma4, configuration et taps DFlash,
contexte par sequence et regles d'acceptation, chargement text-only, PLE CPU
optionnel et comptabilite memoire, correction des templates/outils, fixtures et
tests natifs. Garder le chargeur portable NVFP4 W4A16 qui dequantifie les poids
avant execution : le format du checkpoint NVIDIA est utile sur plusieurs
backends et ne se reduit pas au backend Pascal.

**RETIRER DU PATCH CONTRIB / ISOLER** : le socle Legacy transitif ; le chemin
[packed NVFP4 conditionne par cuda-legacy-fp4](https://github.com/haricot/mistral.rs/blob/9ff03fa63e624cf919108480df9b6ff3b1f4f4b2/mistralrs-core/src/speculative/dflash.rs#L2100),
la selection `preserve_packed` et la reservation memoire specifique de
[external_packed_weight_size_in_bytes](https://github.com/haricot/mistral.rs/blob/9ff03fa63e624cf919108480df9b6ff3b1f4f4b2/mistralrs-core/src/speculative/config.rs#L174)
doivent etre examines ensemble. Retirer seulement le cfg ou seulement
l'estimation risquerait de changer le chemin actif et de sous-estimer la VRAM.
Un backend packed moderne eventuel doit etre selectionne par ses capacites et
qualifie separement ; il ne doit pas disparaitre avec un retrait mecanique du
code SM61.

**CONSERVER MALGRE LE NOM** : `legacy_type` dans la lecture des parametres RoPE
et `legacy_gemma4_tools` concernent des anciens formats de configuration ou de
templates. Ils ne designent pas la branche Legacy ni une limite Pascal.
Le README et `nemotron-bolt/sources.json` historiques doivent etre realignes
avec les references reellement choisies pour la nouvelle contribution.

> Commentaire de revue : separer le support fonctionnel Bolt/DFlash de son
> adaptation packed SM61, conserver le chargeur NVFP4 portable et ses garde-fous,
> et garder les estimations memoire coherentes avec le backend retenu. Le socle
> complet Legacy n'est pas un prerequis fonctionnel acceptable pour cette PR.

### UQFF draft

**CONSERVER** : [plan_draft_uqff](https://github.com/haricot/mistral.rs/blob/53aefbde829a0e8576e3e6b89c9f4da814772bf0/mistralrs-core/src/speculative/dflash_uqff.rs#L30),
le contrat de geometrie/taps/head, la validation Q8_0, les fixtures et la
commande CLI non destructive `uqff draft-plan`.

**PREREQUIS A ISOLER** : cette branche importe un ancien Bolt/DFlash complet,
avec des morceaux de Legacy, meme si le tip Legacy actuel n'est pas un ancetre.
Extraire le plan sur une contribution Bolt/DFlash nettoyee, ou sur une API
reduite de configuration de draft. Ne pas supprimer DFlashConfig en laissant
le preflight dependre d'un symbole disparu. Retirer l'ancien manifeste de
promotion herite et la recette Pascal du futur patch contrib.

**REQUALIFIER** : le [README declare explicitement les limites](https://github.com/haricot/mistral.rs/blob/53aefbde829a0e8576e3e6b89c9f4da814772bf0/uqff-draft/README.md#L15)
ne decrit aucun writer/reloader UQFF draft operationnel. Cette branche fournit
un plan, pas un export Q8_0 fonctionnel. Aucun nettoyage ne doit transformer
ce statut en promesse d'export.

> Commentaire de revue : contribuer d'abord le preflight de draft avec son
> prerequis DFlash minimal, retirer le socle Bolt/Legacy importe sans rapport,
> et conserver le statut plan-only jusqu'a implementation et validation du
> writer et du reloader.

### EmbeddingGemma 2

**CONSERVER** : l'adaptateur local, les dimensions Matryoshka, les identifiants
de tokens, la politique explicite Query/Passage et l'outil de comparaison de
parite. Aucun merge du Legacy actuel n'a ete trouve dans cette branche, et
aucun des deltas Legacy compares n'est recopie a l'identique.

**PREREQUIS A ISOLER** : [le modele provient de Candle Extended](https://github.com/haricot/mistral.rs/blob/641dd2541406476203c286c8d4cd5e431951c103/mistralrs-embeddinggemma2/src/lib.rs#L12)
au SHA `66c4651...`. Le basculement global des packages Candle et de cudaforge
ne peut pas etre simplement efface en gardant cet import. Fournir une
contribution Candle limitee a EmbeddingGemma 2, ou demontrer la presence de
l'API sur la dependance cible, avant d'adapter les pins.

**REQUALIFIER** : le [contrat natif](https://github.com/haricot/mistral.rs/blob/641dd2541406476203c286c8d4cd5e431951c103/mistral-integration/embeddinggemma2-native-contract.md#L3)
reste DESIGN / NOT IMPLEMENTED. La standalone n'ajoute pas encore la famille
au builder et a `/v1/embeddings`. Le F32, la sequence unique et les limitations
de parite restent un perimetre declare ; ce ne sont pas des suppressions Legacy.

> Commentaire de revue : garder le petit adaptateur et les outils de parite,
> isoler le prerequis Candle EmbeddingGemma 2 du reste d'Extended, puis choisir
> explicitement entre une contribution d'adaptateur experimental et une
> integration native complete. Ne pas annoncer l'API native deja implementee.

### CPU-MoE

**CONSERVER** : baseline CPU opt-in, cache Q4K resident par expert, extraction
des projections, workers CPU/prefetch, limites de residence et chemins de
repli. Le delta est directement fonde sur master `095991a...`, sans modification
de Cargo.toml/Cargo.lock, sans merge Legacy et sans dependance ASD declaree.
Le mot `legacy` deja present dans un alias de backend du master n'est pas un
nouvel import a supprimer.

**AJUSTER LA CI** : [le gate courant execute cargo fmt avant de compiler](https://github.com/haricot/mistral.rs/blob/9e39c6c219bc390038e532ddcacec6043d5b8957/.github/workflows/cpu-moe-qualification.yml#L30)
et conserve son patch. Une future preuve de contribution doit compiler l'arbre
exact annonce : utiliser fmt --check, committer les corrections en amont, et
utiliser les commandes Cargo verrouillees adaptees. La doc annonce aussi que
l'overlap CUDA reel et les performances restent a mesurer ; les workers
concurrents ne prouvent pas seuls un recouvrement copies/kernels.

> Commentaire de revue : aucun retrait Legacy identifie dans ce delta ; garder
> le cache et les workers generiques, garantir une CI sur le SHA exact, puis
> qualifier la parite et les ressources sur modele reel avant promotion.

### Streaming des shards UQFF

**CONSERVER** : le delta cible dans
[UqffShardStream](https://github.com/haricot/mistral.rs/blob/c00fc25f6022cddf2f179388b820c441cc36d7a4/mistralrs-core/src/pipeline/isq.rs#L766),
les shards alignes sur les couches, l'ecriture deterministe, la duree de vie
du resultat ISQ pendant la serialisation et les tests de roundtrip. Aucun
socle Legacy ou changement de dependance n'est introduit par ce commit.

**NE PAS SURPROMETTRE** : 512 MiB est une cible de payload, pas un plafond dur
de RAM. Une couche indivisible peut depasser cette taille ; la reserve ISQ
n'est pas une borne sur tous les resultats en attente ou la memoire du processus.

> Commentaire de revue : cette branche est deja un bon candidat a une
> contribution generique. Rebaser le delta sur le master cible, conserver les
> tests et annoncer les limites de memoire exactes ; aucun nettoyage Legacy
> fonctionnel n'a ete identifie.

## Simulations, limites et ordre recommande

Simulations git merge-tree sur le master epingle, sans publication de resolution :

| Branche ancienne | Resultat textuel contre master |
| --- | --- |
| UQFF streaming | Sans conflit textuel ; cela ne prouve pas la compilation |
| EmbeddingGemma 2 | Conflits Cargo.toml et Cargo.lock |
| UQFF draft | 7 chemins en conflit, dont vector_fp8/ops.rs supprime dans master |

Ne pas ressusciter vector_fp8/ops.rs automatiquement pour resoudre le dernier
cas. Les anciennes BF16/FP8/FP4 sont des donneurs historiques ; utiliser les
versions v2 pour le travail de format et leur garder une propriete explicite.

Ordre propose : UQFF streaming et CPU-MoE pour les contributions deja ciblees ;
DSpark pour l'extraction du premier gros import ; Bolt/DFlash ensuite, puis le
preflight UQFF draft qui en depend. ASD et EmbeddingGemma 2 necessitent d'abord
un contrat de dependance Candle propre. Les travaux peuvent etre prepares sans
modifier le candidat Legacy V3 actuellement en attente de qualification.

Cette revue couvre les sept autres standalones et releve les six heads de
format (anciens et v2) pour contexte. Elle ne constitue pas un audit complet
de Candle, une nouvelle compilation des sept branches, une qualification GPU,
une creation de branches contrib fonctionnelles ou une autorisation de
suppression globale. Les fichiers sources et les recettes restent inchanges.
Une extraction future devra figer ses propres sources, lock, features et
preuves ; les anciens resultats ne qualifient pas automatiquement le delta nettoye.
