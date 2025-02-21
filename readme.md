VQCFL
Vector Quantization-Based Clustered Federated Learning with Global Feature Anchors for Improved Representation and Generalization

Clustered Federated Learning (CFL) tackles the challenge of heterogeneous data in federated learning by tailoring models for different client groups. However, existing CFL methods overly rely on indirect metrics, such as model parameters, gradient information, or loss function values, for client clustering strategies. These approaches fail to fully capture the diversity and inherent characteristics of client data distributions, resulting in inaccurate representation of client data features. To address this issue, we propose a novel CFL framework called Vector Quantization-Based CFL (VQCFL). First, we introduce a Vector Quantization Network (VQNet), which effectively captures the intrinsic structure of client data by mapping the local data's feature space into discrete feature dictionary vectors. Additionally, to prevent the drift in the feature dictionary vectors, we propose a global feature anchor strategy that aligns feature dictionary vectors across clients, ensuring consistent updates within the same feature space. Furthermore, we present a novel cross-cluster knowledge-sharing mechanism that integrates feature information from different clusters through global aggregation of feature dictionary vectors. This mechanism, combined with a personalized cross-cluster classifier weight adjustment strategy, significantly improves the model's generalization performance when handling mixed data heterogeneity. Experimental results under various settings demonstrate that VQCFL achieves better local personalization and global generalization performance.
