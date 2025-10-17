import numpy as np
from hdbscan import HDBSCAN
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from qdrant_service import fetch_all_embeddings
import logging
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SpaCy modelini global olarak yükle
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    logger.warning("en_core_web_lg not found, trying en_core_web_sm")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.error("No spaCy model found. Install with: python -m spacy download en_core_web_sm")
        nlp = None


# class TrendClusterAnalyzer:
#     def __init__(
#         self,
#         min_cluster_size: int = 4,  
#         min_samples: int = 2,  
#         metric: str = "cosine",
#         time_decay_days: int = 7,
#         cluster_selection_epsilon: float = 0.20,
#     ):
#         self.min_cluster_size = min_cluster_size
#         self.min_samples = min_samples
#         self.metric = metric
#         self.time_decay_days = time_decay_days
#         self.cluster_selection_epsilon = cluster_selection_epsilon
#         self.scaler = StandardScaler()
#         self.clusterer = None

#     def cluster_with_time_weight(self, vectors_np, payloads, enable_time_weight=True):
#         """Optimize edilmiş zaman ağırlıklı HDBSCAN"""
        
#         logger.info(f"🔍 Clustering {len(vectors_np)} vectors...")

#         # 1. Normalize
#         norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
#         vectors_normalized = vectors_np / np.clip(norms, 1e-8, None)

#         # 2. Zaman ağırlıkları - daha agresif
#         if enable_time_weight:
#             time_weights = self._calculate_time_weights(payloads)
#             # Karekök yerine doğrudan uygula (daha güçlü etki)
#             vectors_weighted = vectors_normalized * time_weights.reshape(-1, 1)
#         else:
#             vectors_weighted = vectors_normalized

#         # 3. Scale
#         vectors_scaled = self.scaler.fit_transform(vectors_weighted)

#         # 4. Distance matrix - daha agresif filtreleme
#         distance_matrix = cosine_distances(vectors_scaled)
        
#         # Outlier'ları daha net ayır
#         distance_threshold = np.percentile(distance_matrix, 80)  # 85->80
#         distance_matrix = np.where(
#             distance_matrix > distance_threshold, 
#             distance_threshold * 1.5,  # 1.2->1.5 daha agresif
#             distance_matrix
#         )

#         # 5. HDBSCAN - daha agresif birleştirme
#         self.clusterer = HDBSCAN(
#             min_cluster_size=self.min_cluster_size,
#             min_samples=self.min_samples,
#             metric="precomputed",
#             cluster_selection_method="eom",
#             cluster_selection_epsilon=self.cluster_selection_epsilon,
#             alpha=1.2,  # 1.0->1.2 daha conservative
#             allow_single_cluster=False,
#         )

#         cluster_labels = self.clusterer.fit_predict(distance_matrix)

#         # 6. Post-processing chain
#         # a) Küçük cluster'ları birleştir
#         cluster_labels = self._merge_small_clusters(
#             cluster_labels, vectors_scaled, min_size=3  # 2->3
#         )
        
#         # b) Benzer cluster'ları birleştir
#         cluster_labels = self._merge_similar_clusters(
#             cluster_labels, vectors_scaled, similarity_threshold=0.85  # Yeni!
#         )
        
#         # c) Outlier'ları yeniden ata
#         cluster_labels = self._reassign_outliers(
#             cluster_labels, vectors_scaled, max_distance=0.35  # Yeni!
#         )

#         # 7. Metrikleri hesapla
#         metrics = self._calculate_clustering_metrics(vectors_scaled, cluster_labels)

#         n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
#         n_outliers = np.sum(cluster_labels == -1)
#         outlier_ratio = n_outliers / len(cluster_labels) * 100

#         logger.info(f"✅ Found {n_clusters} clusters")
#         logger.info(f"📊 Outliers: {n_outliers} ({outlier_ratio:.1f}%)")
#         logger.info(f"📈 Silhouette Score: {metrics.get('silhouette', 'N/A'):.3f}")
#         logger.info(f"📉 Davies-Bouldin: {metrics.get('davies_bouldin', 'N/A'):.3f}")

#         # 8. Organize
#         clusters = self._organize_clusters(cluster_labels, payloads)
#         self._log_cluster_stats(clusters)

#         return cluster_labels, clusters, metrics

#     def _calculate_time_weights(self, payloads):
#         """Daha agresif time decay"""
#         now = datetime.now()
#         weights = []
        
#         for p in payloads:
#             pub_date = p.get("published_at")
#             try:
#                 if isinstance(pub_date, str):
#                     pub_date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                
#                 days_old = (now - pub_date.replace(tzinfo=None)).days
                
#                 # Daha steep decay curve
#                 if days_old <= 1:
#                     w = 1.0  # Bugün/dün
#                 elif days_old <= 3:
#                     w = 0.9  # Son 3 gün
#                 elif days_old <= 7:
#                     w = 0.7  # Son hafta
#                 elif days_old <= 14:
#                     w = 0.4  # 2 hafta
#                 else:
#                     w = 0.2  # Eski
                
#                 weights.append(w)
                
#             except Exception as e:
#                 logger.debug(f"Date parse error: {e}")
#                 weights.append(0.5)
                
#         return np.array(weights)

#     def _merge_similar_clusters(self, labels, vectors, similarity_threshold=0.85):
#         """Birbirine çok benzeyen cluster'ları birleştir"""
#         unique_labels = np.unique(labels[labels >= 0])
        
#         if len(unique_labels) < 2:
#             return labels
        
#         # Her cluster'ın centroid'ini hesapla
#         centroids = {}
#         for label in unique_labels:
#             mask = labels == label
#             centroids[label] = vectors[mask].mean(axis=0)
        
#         # Similarity matrix
#         centroid_vectors = np.array([centroids[l] for l in unique_labels])
#         similarity_matrix = cosine_similarity(centroid_vectors)
        
#         # Merge mapping
#         merge_map = {label: label for label in unique_labels}
        
#         for i, label_i in enumerate(unique_labels):
#             for j, label_j in enumerate(unique_labels):
#                 if i >= j:
#                     continue
                
#                 if similarity_matrix[i, j] > similarity_threshold:
#                     # Küçük cluster'ı büyük cluster'a birleştir
#                     size_i = np.sum(labels == label_i)
#                     size_j = np.sum(labels == label_j)
                    
#                     if size_i > size_j:
#                         merge_map[label_j] = merge_map[label_i]
#                     else:
#                         merge_map[label_i] = merge_map[label_j]
        
#         # Apply merges
#         new_labels = labels.copy()
#         for old_label, new_label in merge_map.items():
#             new_labels[labels == old_label] = new_label
        
#         # Re-index clusters sequentially
#         unique_new = np.unique(new_labels[new_labels >= 0])
#         label_mapping = {old: new for new, old in enumerate(unique_new)}
#         label_mapping[-1] = -1
        
#         final_labels = np.array([label_mapping[l] for l in new_labels])
        
#         n_merged = len(unique_labels) - len(unique_new)
#         if n_merged > 0:
#             logger.info(f"🔗 Merged {n_merged} similar clusters")
        
#         return final_labels

#     def _reassign_outliers(self, labels, vectors, max_distance=0.35):
#         """Outlier'ları yakın cluster'lara ata"""
#         outlier_mask = labels == -1
#         n_outliers = np.sum(outlier_mask)
        
#         if n_outliers == 0:
#             return labels
        
#         unique_labels = np.unique(labels[labels >= 0])
#         if len(unique_labels) == 0:
#             return labels
        
#         # Her cluster'ın centroid'i
#         centroids = {}
#         for label in unique_labels:
#             mask = labels == label
#             centroids[label] = vectors[mask].mean(axis=0)
        
#         # Her outlier için en yakın cluster'ı bul
#         new_labels = labels.copy()
#         reassigned = 0
        
#         outlier_indices = np.where(outlier_mask)[0]
#         for idx in outlier_indices:
#             outlier_vec = vectors[idx].reshape(1, -1)
            
#             min_dist = float('inf')
#             closest_cluster = -1
            
#             for label, centroid in centroids.items():
#                 dist = cosine_distances(outlier_vec, centroid.reshape(1, -1))[0, 0]
#                 if dist < min_dist:
#                     min_dist = dist
#                     closest_cluster = label
            
#             # Eğer yeterince yakınsa ata
#             if min_dist < max_distance:
#                 new_labels[idx] = closest_cluster
#                 reassigned += 1
        
#         if reassigned > 0:
#             logger.info(f"🎯 Reassigned {reassigned}/{n_outliers} outliers to clusters")
        
#         return new_labels

#     def _merge_small_clusters(self, labels, vectors, min_size=3):
#         """Küçük cluster'ları birleştir"""
#         unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        
#         for label, count in zip(unique_labels, counts):
#             if count < min_size:
#                 mask = labels == label
#                 cluster_centroid = vectors[mask].mean(axis=0).reshape(1, -1)
                
#                 # Diğer büyük cluster'lara uzaklık
#                 large_clusters = unique_labels[counts >= min_size]
#                 if len(large_clusters) == 0:
#                     labels[mask] = -1  # Outlier yap
#                     continue
                
#                 min_dist = float('inf')
#                 closest_cluster = -1
                
#                 for large_label in large_clusters:
#                     large_mask = labels == large_label
#                     large_centroid = vectors[large_mask].mean(axis=0).reshape(1, -1)
#                     dist = cosine_distances(cluster_centroid, large_centroid)[0, 0]
                    
#                     if dist < min_dist:
#                         min_dist = dist
#                         closest_cluster = large_label
                
#                 # Threshold check
#                 if min_dist < 0.4:
#                     labels[mask] = closest_cluster
#                 else:
#                     labels[mask] = -1  # Çok uzaksa outlier yap
                    
#         return labels

#     def _calculate_clustering_metrics(self, vectors, labels):
#         """Calculate clustering quality metrics"""
#         metrics = {}
#         mask = labels >= 0
        
#         if np.sum(mask) > 1 and len(np.unique(labels[mask])) > 1:
#             try:
#                 metrics["silhouette"] = silhouette_score(
#                     vectors[mask], labels[mask], metric="cosine"
#                 )
#                 metrics["davies_bouldin"] = davies_bouldin_score(
#                     vectors[mask], labels[mask]
#                 )
                
#                 from sklearn.metrics import calinski_harabasz_score
#                 metrics["calinski_harabasz"] = calinski_harabasz_score(
#                     vectors[mask], labels[mask]
#                 )
                
#             except Exception as e:
#                 logger.warning(f"Metric calculation error: {e}")
        
#         metrics["outlier_count"] = int(np.sum(labels == -1))
#         metrics["outlier_ratio"] = float(np.sum(labels == -1) / len(labels))
#         metrics["n_clusters"] = len(np.unique(labels[labels >= 0]))
        
#         return metrics

#     def _organize_clusters(self, labels, payloads):
#         """Organize documents into clusters with metadata"""
#         clusters = defaultdict(list)
        
#         for label, payload in zip(labels, payloads):
#             enriched = payload.copy()
#             enriched["cluster_label"] = int(label)
#             clusters[int(label)].append(enriched)
        
#         sorted_clusters = dict(
#             sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
#         )
        
#         return sorted_clusters

#     def _log_cluster_stats(self, clusters):
#         """Log detailed cluster statistics"""
#         if -1 in clusters:
#             non_outlier_clusters = {k: v for k, v in clusters.items() if k != -1}
#         else:
#             non_outlier_clusters = clusters
        
#         if non_outlier_clusters:
#             sizes = [len(v) for v in non_outlier_clusters.values()]
#             logger.info(f"📊 Cluster sizes - Min: {min(sizes)}, Max: {max(sizes)}, "
#                        f"Mean: {np.mean(sizes):.1f}, Median: {np.median(sizes):.1f}")
            
#             # Distribution
#             small = sum(1 for s in sizes if s < 5)
#             medium = sum(1 for s in sizes if 5 <= s < 15)
#             large = sum(1 for s in sizes if s >= 15)
#             logger.info(f"📈 Distribution - Small(<5): {small}, Medium(5-15): {medium}, Large(≥15): {large}")
            
#             # Top 5 largest clusters
#             top_5 = sorted(non_outlier_clusters.items(), 
#                           key=lambda x: len(x[1]), reverse=True)[:5]
#             logger.info(f"🏆 Top 5 clusters: {[f'C{k}({len(v)})' for k, v in top_5]}")

#     def get_cluster_summary(self, clusters):
#         """Get a summary of all clusters"""
#         summary = []
        
#         for cluster_id, docs in clusters.items():
#             if cluster_id == -1:
#                 continue
                
#             dates = []
#             for doc in docs:
#                 try:
#                     pub_date = doc.get("published_at")
#                     if isinstance(pub_date, str):
#                         pub_date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
#                     dates.append(pub_date)
#                 except:
#                     continue
            
#             avg_date = None
#             if dates:
#                 avg_timestamp = np.mean([d.timestamp() for d in dates])
#                 avg_date = datetime.fromtimestamp(avg_timestamp)
            
#             summary.append({
#                 "cluster_id": cluster_id,
#                 "size": len(docs),
#                 "avg_date": avg_date,
#                 "docs": docs
#             })
        
#         summary.sort(key=lambda x: x["size"], reverse=True)
#         return summary

class TrendClusterAnalyzer:
    def __init__(
        self,
        min_cluster_size: int = 5,  
        min_samples: int = 2,
        time_decay_days: int = 10,
        cluster_selection_epsilon: float = 0.22, 
        outlier_reassignment_distance: float = 0.38,  
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.time_decay_days = time_decay_days
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.outlier_reassignment_distance = outlier_reassignment_distance
        self.scaler = StandardScaler()
        self.clusterer = None

    def cluster_with_time_weight(self, vectors_np, payloads, enable_time_weight=True):
        """Zaman ağırlıklı ve cosine distance ile HDBSCAN clustering"""
        logger.info(f"🔍 Clustering {len(vectors_np)} vectors...")

        # 1. Normalize
        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        vectors_normalized = vectors_np / np.clip(norms, 1e-8, None)

        # 2. Zaman ağırlıkları
        if enable_time_weight:
            time_weights = self._calculate_time_weights(payloads)
            vectors_weighted = vectors_normalized * time_weights.reshape(-1, 1)
        else:
            vectors_weighted = vectors_normalized

        # 3. Scale
        vectors_scaled = self.scaler.fit_transform(vectors_weighted)

        # 4. Cosine distance matrix
        distance_matrix = cosine_distances(vectors_scaled)

        # 5. Threshold softening
        distance_threshold = np.percentile(distance_matrix, 82)
        distance_matrix = np.where(
            distance_matrix > distance_threshold, 
            distance_threshold * 1.3,
            distance_matrix
        )

        # 6. HDBSCAN clustering
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="precomputed",
            cluster_selection_method="eom",
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            alpha=1.0,
            allow_single_cluster=False,
        )
        cluster_labels = self.clusterer.fit_predict(distance_matrix)

        # 7. Post-processing
        cluster_labels = self._merge_small_clusters(cluster_labels, vectors_scaled, min_size=3)
        cluster_labels = self._reassign_outliers(cluster_labels, vectors_scaled, max_distance=self.outlier_reassignment_distance)
        cluster_labels = self._merge_similar_clusters(cluster_labels, vectors_scaled, similarity_threshold=0.85)
        cluster_labels = self._reassign_outliers(cluster_labels, vectors_scaled, max_distance=self.outlier_reassignment_distance + 0.05)
        cluster_labels = self._reassign_outliers_soft(cluster_labels, vectors_scaled, payloads, max_distance=0.50)

        # 8. Metrics
        metrics = self._calculate_clustering_metrics(vectors_scaled, cluster_labels)
        n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
        n_outliers = np.sum(cluster_labels == -1)
        outlier_ratio = n_outliers / len(cluster_labels) * 100

        logger.info(f"✅ Found {n_clusters} clusters")
        logger.info(f"📊 Outliers: {n_outliers} ({outlier_ratio:.1f}%)")
        logger.info(f"📈 Silhouette Score: {metrics.get('silhouette', 'N/A'):.3f}")
        logger.info(f"📉 Davies-Bouldin: {metrics.get('davies_bouldin', 'N/A'):.3f}")

        clusters = self._organize_clusters(cluster_labels, payloads)
        self._log_cluster_stats(clusters)

        return cluster_labels, clusters, metrics

    def _calculate_time_weights(self, payloads):
        now = datetime.now()
        weights = []
        for p in payloads:
            pub_date = p.get("published_at")
            try:
                if isinstance(pub_date, str):
                    pub_date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                days_old = (now - pub_date.replace(tzinfo=None)).days
                if days_old <= 1:
                    w = 1.0
                elif days_old <= 3:
                    w = 0.9
                elif days_old <= 7:
                    w = 0.7
                elif days_old <= 14:
                    w = 0.4
                else:
                    w = 0.2
                weights.append(w)
            except:
                weights.append(0.5)
        return np.array(weights)

    def _merge_similar_clusters(self, labels, vectors, similarity_threshold=0.85):
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) < 2:
            return labels
        centroids = {l: vectors[labels==l].mean(axis=0) for l in unique_labels}
        centroid_vectors = np.array([centroids[l] for l in unique_labels])
        similarity_matrix = cosine_similarity(centroid_vectors)
        merge_map = {label: label for label in unique_labels}
        for i, label_i in enumerate(unique_labels):
            for j, label_j in enumerate(unique_labels):
                if i >= j: continue
                if similarity_matrix[i,j] > similarity_threshold:
                    size_i = np.sum(labels==label_i)
                    size_j = np.sum(labels==label_j)
                    if size_i > size_j:
                        merge_map[label_j] = merge_map[label_i]
                    else:
                        merge_map[label_i] = merge_map[label_j]
        new_labels = labels.copy()
        for old_label, new_label in merge_map.items():
            new_labels[labels==old_label] = new_label
        unique_new = np.unique(new_labels[new_labels>=0])
        label_mapping = {old: new for new, old in enumerate(unique_new)}
        label_mapping[-1] = -1
        final_labels = np.array([label_mapping[l] for l in new_labels])
        n_merged = len(unique_labels) - len(unique_new)
        if n_merged > 0:
            logger.info(f"🔗 Merged {n_merged} similar clusters")
        return final_labels

    def _reassign_outliers_soft(self, labels, vectors, payloads, max_distance=0.50):
        outlier_mask = labels == -1
        n_outliers = np.sum(outlier_mask)
        if n_outliers == 0:
            return labels
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) == 0:
            return labels
        cluster_info = {}
        for label in unique_labels:
            mask = labels==label
            cluster_info[label] = {'centroid': vectors[mask].mean(axis=0), 'size': np.sum(mask)}
        new_labels = labels.copy()
        reassigned = 0
        outlier_indices = np.where(outlier_mask)[0]
        for idx in outlier_indices:
            outlier_vec = vectors[idx].reshape(1,-1)
            scores = {}
            for label, info in cluster_info.items():
                dist = cosine_distances(outlier_vec, info['centroid'].reshape(1,-1))[0,0]
                size_bonus = min(0.05, info['size']/1000)
                adjusted_dist = dist - size_bonus
                if adjusted_dist < max_distance:
                    scores[label] = adjusted_dist
            if scores:
                best_cluster = min(scores, key=scores.get)
                new_labels[idx] = best_cluster
                reassigned += 1
        if reassigned > 0:
            logger.info(f"🎯 SOFT reassigned {reassigned}/{n_outliers} remaining outliers")
        return new_labels

    def _reassign_outliers(self, labels, vectors, max_distance=0.40):
        outlier_mask = labels == -1
        n_outliers = np.sum(outlier_mask)
        if n_outliers == 0:
            return labels
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) == 0:
            return labels
        centroids = {label: vectors[labels==label].mean(axis=0) for label in unique_labels}
        new_labels = labels.copy()
        reassigned = 0
        outlier_indices = np.where(outlier_mask)[0]
        for idx in outlier_indices:
            outlier_vec = vectors[idx].reshape(1,-1)
            min_dist = float('inf')
            closest_cluster = -1
            for label, centroid in centroids.items():
                dist = cosine_distances(outlier_vec, centroid.reshape(1,-1))[0,0]
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = label
            if min_dist < max_distance:
                new_labels[idx] = closest_cluster
                reassigned += 1
        if reassigned > 0:
            logger.info(f"🎯 Reassigned {reassigned}/{n_outliers} outliers to clusters")
        return new_labels

    def _merge_small_clusters(self, labels, vectors, min_size=3):
        unique_labels, counts = np.unique(labels[labels>=0], return_counts=True)
        for label, count in zip(unique_labels, counts):
            if count < min_size:
                mask = labels==label
                cluster_centroid = vectors[mask].mean(axis=0).reshape(1,-1)
                large_clusters = unique_labels[counts>=min_size]
                if len(large_clusters)==0:
                    labels[mask]=-1
                    continue
                min_dist = float('inf')
                closest_cluster=-1
                for large_label in large_clusters:
                    large_centroid = vectors[labels==large_label].mean(axis=0).reshape(1,-1)
                    dist = cosine_distances(cluster_centroid, large_centroid)[0,0]
                    if dist < min_dist:
                        min_dist=dist
                        closest_cluster=large_label
                if min_dist < 0.4:
                    labels[mask]=closest_cluster
                else:
                    labels[mask]=-1
        return labels

    def _calculate_clustering_metrics(self, vectors, labels):
        metrics = {}
        mask = labels>=0
        if np.sum(mask)>1 and len(np.unique(labels[mask]))>1:
            try:
                metrics["silhouette"] = silhouette_score(vectors[mask], labels[mask], metric="cosine")
                metrics["davies_bouldin"] = davies_bouldin_score(vectors[mask], labels[mask])
            except Exception as e:
                logger.warning(f"Metric calculation error: {e}")
        metrics["outlier_count"] = int(np.sum(labels==-1))
        metrics["outlier_ratio"] = float(np.sum(labels==-1)/len(labels))
        metrics["n_clusters"] = len(np.unique(labels[labels>=0]))
        return metrics

    def _organize_clusters(self, labels, payloads):
        clusters = defaultdict(list)
        for label, payload in zip(labels, payloads):
            enriched = payload.copy()
            enriched["cluster_label"]=int(label)
            clusters[int(label)].append(enriched)
        sorted_clusters = dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True))
        return sorted_clusters

    def _log_cluster_stats(self, clusters):
        non_outlier_clusters = {k:v for k,v in clusters.items() if k!=-1}
        if non_outlier_clusters:
            sizes = [len(v) for v in non_outlier_clusters.values()]
            logger.info(f"📊 Cluster sizes - Min: {min(sizes)}, Max: {max(sizes)}, Mean: {np.mean(sizes):.1f}, Median: {np.median(sizes):.1f}")
            small = sum(1 for s in sizes if s<5)
            medium = sum(1 for s in sizes if 5<=s<15)
            large = sum(1 for s in sizes if s>=15)
            logger.info(f"📈 Distribution - Small(<5): {small}, Medium(5-15): {medium}, Large(≥15): {large}")
            top_5 = sorted(non_outlier_clusters.items(), key=lambda x: len(x[1]), reverse=True)[:5]
            logger.info(f"🏆 Top 5 clusters: {[f'C{k}({len(v)})' for k,v in top_5]}")

    def get_cluster_summary(self, clusters):
        summary=[]
        for cluster_id, docs in clusters.items():
            if cluster_id==-1:
                continue
            dates=[]
            for doc in docs:
                try:
                    pub_date = doc.get("published_at")
                    if isinstance(pub_date,str):
                        pub_date = datetime.fromisoformat(pub_date.replace("Z","+00:00"))
                    dates.append(pub_date)
                except:
                    continue
            avg_date=None
            if dates:
                avg_timestamp=np.mean([d.timestamp() for d in dates])
                avg_date=datetime.fromtimestamp(avg_timestamp)
            summary.append({
                "cluster_id":cluster_id,
                "size":len(docs),
                "avg_date":avg_date,
                "docs":docs
            })
        summary.sort(key=lambda x: x["size"], reverse=True)
        return summary

class OptimizedTrendClusterAnalyzer(TrendClusterAnalyzer):
    def __init__(self):
        super().__init__(
            min_cluster_size=10,      # Küçük cluster'ları azalt
            min_samples=3,            # Daha sıkı noise kontrolü
            metric="cosine",
            time_decay_days=5,        # Daha agresif time decay
            cluster_selection_epsilon=0.30,  # Küçük farkları cluster yap
            outlier_reassignment_distance=0.35  # Outlier'lar daha kolay reassigned
        )

    def cluster_with_time_weight(self, vectors_np, payloads, enable_time_weight=True):
        logger.info(f"🔍 Clustering {len(vectors_np)} vectors...")

        # 1. Normalize
        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        vectors_normalized = vectors_np / np.clip(norms, 1e-8, None)

        # 2. Zaman ağırlığı - daha agresif
        if enable_time_weight:
            time_weights = self._calculate_time_weights(payloads)
            vectors_weighted = vectors_normalized * time_weights.reshape(-1, 1)
        else:
            vectors_weighted = vectors_normalized

        # 3. Scale
        vectors_scaled = self.scaler.fit_transform(vectors_weighted)

        # 4. Distance matrix + smoothing
        distance_matrix = cosine_distances(vectors_scaled)
        dist_threshold = np.percentile(distance_matrix, 80)
        distance_matrix = np.where(distance_matrix > dist_threshold,
                                   dist_threshold * 1.2,
                                   distance_matrix)

        # 5. HDBSCAN
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="precomputed",
            cluster_selection_method="eom",
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            alpha=1.0,
            allow_single_cluster=False
        )
        cluster_labels = self.clusterer.fit_predict(distance_matrix)

        # 6. Post-processing zinciri
        cluster_labels = self._merge_small_clusters(cluster_labels, vectors_scaled, min_size=5)
        cluster_labels = self._reassign_outliers(cluster_labels, vectors_scaled, max_distance=self.outlier_reassignment_distance)
        cluster_labels = self._merge_similar_clusters(cluster_labels, vectors_scaled, similarity_threshold=0.82)
        cluster_labels = self._reassign_outliers(cluster_labels, vectors_scaled, max_distance=self.outlier_reassignment_distance+0.05)
        cluster_labels = self._reassign_outliers_soft(cluster_labels, vectors_scaled, payloads, max_distance=0.45)

        # 7. Metrikler
        metrics = self._calculate_clustering_metrics(vectors_scaled, cluster_labels)

        n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
        n_outliers = np.sum(cluster_labels == -1)
        outlier_ratio = n_outliers / len(cluster_labels) * 100

        logger.info(f"✅ Found {n_clusters} clusters")
        logger.info(f"📊 Outliers: {n_outliers} ({outlier_ratio:.1f}%)")
        logger.info(f"📈 Silhouette Score: {metrics.get('silhouette', 'N/A'):.3f}")
        logger.info(f"📉 Davies-Bouldin: {metrics.get('davies_bouldin', 'N/A'):.3f}")

        clusters = self._organize_clusters(cluster_labels, payloads)
        self._log_cluster_stats(clusters)

        return cluster_labels, clusters, metrics

class AdvancedKeywordExtractor:
    """Gelişmiş keyword extraction sistemi"""
    
    def __init__(self, language: str = 'english'):
        self.language = language
    
    def extract_cluster_trends(
        self, 
        clusters: Dict[int, List[dict]],
        top_n: int = 5,
        use_ngrams: bool = True
    ) -> Dict[int, Dict]:
        """
        Her cluster için trend analizı yap
        """
        cluster_analysis = {}
        
        for label, news_list in clusters.items():
            if label == -1:  # Skip outliers
                continue
            
            if len(news_list) < 2:
                continue
            
            # Başlık ve içeriklerden keyword çıkar
            titles = [n.get('title', '') for n in news_list]
            contents = [n.get('content', '') for n in news_list]
            
            # TF-IDF ile keyword extraction
            title_keywords = self._extract_keywords(
                titles, 
                top_n=top_n,
                use_ngrams=use_ngrams
            )
            
            # Entity extraction - spaCy varsa kullan, yoksa basit yöntem
            entities = self._extract_entities(titles)
            
            # Zaman analizi
            time_analysis = self._analyze_temporal_distribution(news_list)
            
            # Source analizi
            source_distribution = self._analyze_sources(news_list)
            
            cluster_analysis[label] = {
                'size': len(news_list),
                'keywords': title_keywords,
                'top_entities': entities[:5],
                'time_span': time_analysis,
                'sources': source_distribution,
                'representative_title': self._get_representative_title(news_list, title_keywords),
                'trend_score': self._calculate_trend_score(news_list, time_analysis)
            }
        
        return cluster_analysis
    
    def _extract_keywords(
        self, 
        texts: List[str], 
        top_n: int = 5,
        use_ngrams: bool = True
    ) -> List[Tuple[str, float]]:
        """TF-IDF ile keyword çıkarımı"""
        if not texts or all(not t for t in texts):
            return []
        
        ngram_range = (1, 2) if use_ngrams else (1, 1)
        
        try:
            vectorizer = TfidfVectorizer(
                stop_words=self.language,
                max_features=100,
                ngram_range=ngram_range,
                min_df=1,
                max_df=0.8
            )
            
            X = vectorizer.fit_transform(texts)
            scores = np.asarray(X.sum(axis=0)).ravel()
            features = vectorizer.get_feature_names_out()
            
            # Keyword-score pairs
            keyword_scores = sorted(
                zip(features, scores),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            return keyword_scores
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []
    
    def _extract_entities(self, texts: List[str]) -> List[str]:
        """
        Entity extraction - spaCy varsa NER kullan, yoksa basit yöntem
        """
        # SpaCy ile entity extraction
        if nlp is not None:
            try:
                return self._extract_entities_spacy(texts)
            except Exception as e:
                logger.warning(f"SpaCy entity extraction failed: {e}, falling back to simple method")
        
        # Basit fallback: büyük harfle başlayan kelimeler
        return self._extract_entities_simple(texts)
    
    def _extract_entities_spacy(self, texts: List[str]) -> List[str]:
        """
        SpaCy ile entity extraction
        """
        entities = []
        for text in texts:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "EVENT", "PRODUCT"}:
                    entities.append(ent.text)
        
        # En sık geçenleri döndür
        counter = Counter(entities)
        return [entity for entity, count in counter.most_common(10)]
    
    def _extract_entities_simple(self, texts: List[str]) -> List[str]:
        """
        Basit entity extraction - büyük harfle başlayan kelimeler
        """
        entities = []
        for text in texts:
            words = text.split()
            # 2+ kelimeden oluşan büyük harfli ifadeler
            i = 0
            while i < len(words):
                if i < len(words) and len(words[i]) > 2 and words[i][0].isupper():
                    entity = words[i]
                    # Ardışık büyük harfli kelimeleri birleştir
                    j = i + 1
                    while j < len(words) and len(words[j]) > 0 and words[j][0].isupper():
                        entity += " " + words[j]
                        j += 1
                    if len(entity.split()) >= 2:  # En az 2 kelimelik entity'ler
                        entities.append(entity)
                    i = j
                else:
                    i += 1
        
        # En sık geçenleri döndür
        counter = Counter(entities)
        return [entity for entity, count in counter.most_common(10)]

    def _analyze_temporal_distribution(self, news_list: List[dict]) -> Dict:
        """Haberlerin zaman dağılımını analiz et"""
        dates = []
        for news in news_list:
            pub_date = news.get('published_at')
            if pub_date:
                try:
                    if isinstance(pub_date, str):
                        pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    dates.append(pub_date.replace(tzinfo=None))
                except Exception as e:
                    logger.warning(f"Date parsing error in temporal analysis: {e}")
                    continue
        
        if not dates:
            return {}
        
        dates.sort()
        time_span_hours = (dates[-1] - dates[0]).total_seconds() / 3600
        
        return {
            'first_seen': dates[0].isoformat(),
            'last_seen': dates[-1].isoformat(),
            'span_hours': round(time_span_hours, 2),
            'velocity': len(dates) / max(time_span_hours, 1)  # news/hour
        }
    
    def _analyze_sources(self, news_list: List[dict]) -> Dict[str, int]:
        """Kaynak dağılımını analiz et"""
        sources = [n.get('source', 'unknown') for n in news_list]
        counter = Counter(sources)
        return dict(counter.most_common(5))
    
    def _get_representative_title(
        self, 
        news_list: List[dict], 
        keywords: List[Tuple[str, float]]
    ) -> str:
        """En temsili başlığı bul"""
        if not keywords:
            return news_list[0].get('title', '')
        
        keyword_set = {kw for kw, _ in keywords}
        
        # En fazla keyword içeren başlık
        best_title = ""
        max_match = 0
        
        for news in news_list:
            title = news.get('title', '').lower()
            match_count = sum(1 for kw in keyword_set if kw.lower() in title)
            if match_count > max_match:
                max_match = match_count
                best_title = news.get('title', '')
        
        return best_title or news_list[0].get('title', '')
    
    def _calculate_trend_score(
        self, 
        news_list: List[dict], 
        time_analysis: Dict
    ) -> float:
        """
        Trend skorunu hesapla (0-100)
        Faktörler: cluster boyutu, yayılma hızı, güncellik
        """
        size_score = min(len(news_list) / 20, 1.0) * 40  # Max 40 puan
        
        velocity = time_analysis.get('velocity', 0)
        velocity_score = min(velocity / 5, 1.0) * 30  # Max 30 puan
        
        # Güncellik skoru
        if time_analysis.get('last_seen'):
            try:
                last_seen = datetime.fromisoformat(time_analysis['last_seen'])
                hours_ago = (datetime.now() - last_seen).total_seconds() / 3600
                recency_score = max(0, (24 - hours_ago) / 24) * 30  # Max 30 puan
            except Exception as e:
                logger.warning(f"Error calculating recency score: {e}")
                recency_score = 0
        else:
            recency_score = 0
        
        return round(size_score + velocity_score + recency_score, 2)


def generate_trend_report(cluster_analysis: Dict[int, Dict]) -> str:
    """Okunabilir trend raporu oluştur"""
    # Trend skoruna göre sırala
    sorted_clusters = sorted(
        cluster_analysis.items(),
        key=lambda x: x[1].get('trend_score', 0),
        reverse=True
    )
    
    report = "\n" + "="*80 + "\n"
    report += "🔥 TRENDING NEWS CLUSTERS REPORT\n"
    report += "="*80 + "\n\n"
    
    for rank, (label, analysis) in enumerate(sorted_clusters[:10], 1):
        report += f"#{rank} Cluster {label} | Trend Score: {analysis['trend_score']:.1f}/100\n"
        report += f"📰 Size: {analysis['size']} articles\n"
        report += f"🏷️  Keywords: {', '.join([kw for kw, _ in analysis['keywords']])}\n"
        
        if analysis.get('top_entities'):
            report += f"👤 Entities: {', '.join(analysis['top_entities'][:3])}\n"
        
        if analysis['time_span']:
            report += f"⏱️  Velocity: {analysis['time_span'].get('velocity', 0):.2f} news/hour\n"
        
        report += f"📌 Representative: {analysis['representative_title'][:80]}...\n"
        report += "-" * 80 + "\n\n"
    
    return report



def normalize_vectors(vectors_np):
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    return vectors_np / np.clip(norms, 1e-8, None)

def grid_search_hdbscan(vectors_np, payloads, param_grid, enable_time_weight=True):
    """
    param_grid = {
        'min_cluster_size': [5, 8, 10],
        'min_samples': [1, 2, 3],
        'cluster_selection_epsilon': [0.15, 0.22, 0.3],
        'outlier_reassignment_distance': [0.35, 0.4, 0.45],
        'time_decay_days': [7, 10, 14]
    }
    """
    results = []
    
    vectors_normalized = normalize_vectors(vectors_np)
    scaler = StandardScaler()
    
    # Time weights
    if enable_time_weight:
        now = np.datetime64('now')
        time_weights = []
        for p in payloads:
            pub_date = p.get("published_at")
            try:
                if isinstance(pub_date, str):
                    pub_date = np.datetime64(pub_date.replace("Z", ""))
                days_old = (now - pub_date).astype(int)
                weight = np.exp(-days_old / 10)  # default decay
                time_weights.append(max(weight, 0.1))
            except:
                time_weights.append(0.5)
        time_weights = np.array(time_weights)
        vectors_weighted = vectors_normalized * time_weights.reshape(-1, 1)
    else:
        vectors_weighted = vectors_normalized
    
    vectors_scaled = scaler.fit_transform(vectors_weighted)
    
    # Grid search
    keys, values = zip(*param_grid.items())
    for combination in product(*values):
        params = dict(zip(keys, combination))
        clusterer = HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            metric='cosine',
            cluster_selection_method='eom',
            cluster_selection_epsilon=params['cluster_selection_epsilon'],
            prediction_data=True
        )
        
        labels = clusterer.fit_predict(vectors_scaled)
        
        outliers = np.sum(labels == -1)
        n_clusters = len(np.unique(labels[labels >= 0]))
        
        if n_clusters > 1:
            try:
                sil_score = silhouette_score(vectors_scaled[labels>=0], labels[labels>=0], metric='cosine')
            except:
                sil_score = float('nan')
        else:
            sil_score = float('nan')
        
        outlier_ratio = outliers / len(labels)
        
        logger.info(f"Params: {params} | Clusters: {n_clusters} | Outliers: {outliers} ({outlier_ratio:.1%}) | Silhouette: {sil_score:.3f}")
        results.append({
            'params': params,
            'n_clusters': n_clusters,
            'outliers': outliers,
            'outlier_ratio': outlier_ratio,
            'silhouette': sil_score
        })
    
    # En iyi skorları döndür
    results_sorted = sorted(results, key=lambda x: (x['silhouette'], -x['outlier_ratio']), reverse=True)
    return results_sorted


def main():
    # 1️⃣ Vektörleri ve payload'ları çek
    vectors, payloads = fetch_all_embeddings()
    vectors = np.array(vectors)

    logger.info(f"✅ Fetched {len(vectors)} vectors from Qdrant")

    # 2️⃣ Grid search için parametreler
    param_grid = {
        'min_cluster_size': [5, 8, 10],
        'min_samples': [1, 2, 3],
        'cluster_selection_epsilon': [0.15, 0.22, 0.3],
        'outlier_reassignment_distance': [0.35, 0.4, 0.45],
        'time_decay_days': [7, 10, 14]
    }

    # 3️⃣ Grid search çalıştır
    logger.info("🔍 Starting HDBSCAN grid search...")
    results = grid_search_hdbscan(vectors, payloads, param_grid, enable_time_weight=True)

    # 4️⃣ En iyi 5 parametre setini yazdır
    logger.info("🏆 Top 5 parameter sets:")
    for i, res in enumerate(results[:5], 1):
        logger.info(f"{i}. Params: {res['params']} | Silhouette: {res['silhouette']:.3f} | "
                    f"Outlier ratio: {res['outlier_ratio']:.1%} | Clusters: {res['n_clusters']}")
    
    # 5️⃣ İstersen buradan seçtiğin parametreyle gerçek clustering yapabilirsin
    best_params = results[0]['params']
    clusterer = HDBSCAN(
        min_cluster_size=best_params['min_cluster_size'],
        min_samples=best_params['min_samples'],
        metric='cosine',
        cluster_selection_method='eom',
        cluster_selection_epsilon=best_params['cluster_selection_epsilon'],
        prediction_data=True
    )
    cluster_labels = clusterer.fit_predict(vectors)
    logger.info("✅ Clustering finished with best parameters")

if __name__ == "__main__":
    main()

