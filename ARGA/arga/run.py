import settings

from clustering import Clustering_Runner
from link_prediction import Link_pred_Runner
from attribute_prediction import Attr_pred_Runner


dataname = 'pubmed'                     # 'cora' or 'citeseer' or 'pubmed'
model = 'arga_ae'                     # 'arga_ae' or 'arga_vae'
task = 'link_prediction'            # 'clustering' or 'link_prediction' or 'attribute_prediction'
n_hop_enable = False                  # True or False                                                                                             # Author: Tonni
hop_count = 0                         # Degree of neighbours. For example, hop_count=1 means info till friends of neighbouring nodes              # Author: Tonni
pred_column = None                       # None or column number                                                                                     # Author: Tonni


settings = settings.get_settings(dataname, model, task, n_hop_enable, hop_count, pred_column)
print("settings:", settings)

if task == 'clustering':
    runner = Clustering_Runner(settings)
elif task == 'link_prediction':
    runner = Link_pred_Runner(settings)
else:                                             # Author: Tonni
    runner = Attr_pred_Runner(settings)           # Author: Tonni


runner.erun()

