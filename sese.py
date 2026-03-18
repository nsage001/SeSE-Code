import numpy as np
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SeSEUncertainty:
    def __init__(
        self,
        sim_threshold=0.7,
        entailment_threshold=0.6,
        device="gpu"
    ):
        self.sim_threshold = sim_threshold
        self.entailment_threshold = entailment_threshold
        self.device = device

        # Embedding model (for fast filtering)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # NLI model (for directed edges)
        self.nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large-mnli"
        ).to(device)

    # ----------------------------------
    # 1. Claim Extraction
    # ----------------------------------
    def extract_claims(self, responses):
        claims = []
        mapping = []

        for i, r in enumerate(responses):
            sentences = [s.strip() for s in r.split(".") if len(s.strip()) > 5]
            for s in sentences:
                claims.append(s)
                mapping.append(i)  # track which response it came from

        return claims, mapping

    # ----------------------------------
    # 2. Fast Similarity Filter
    # ----------------------------------
    def compute_similarity(self, claims):
        embeddings = self.embedder.encode(claims)
        sim_matrix = cosine_similarity(embeddings)
        return sim_matrix

    # ----------------------------------
    # 3. Entailment Score (Directed)
    # ----------------------------------
    def entailment_score(self, premise, hypothesis):
        inputs = self.nli_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)

        # MNLI: [contradiction, neutral, entailment]
        return probs[0, 2].item()

    # ----------------------------------
    # 4. Build Directed Graph
    # ----------------------------------
    def build_graph(self, claims):
        G = nx.DiGraph()

        for i, c in enumerate(claims):
            G.add_node(i, text=c)

        sim_matrix = self.compute_similarity(claims)

        n = len(claims)

        for i in range(n):
            for j in range(n):

                if i == j:
                    continue

                # fast prune
                if sim_matrix[i][j] < self.sim_threshold:
                    continue

                score = self.entailment_score(claims[i], claims[j])

                if score > self.entailment_threshold:
                    G.add_edge(i, j, weight=score)

        return G

    # ----------------------------------
    # 5. Structural Entropy (SeSE)
    # ----------------------------------
    def structural_entropy(self, G):

        # Use weighted out-degree (information flow)
        weights = []

        for node in G.nodes():
            w = sum([G[node][nbr]['weight'] for nbr in G.successors(node)])
            weights.append(w)

        weights = np.array(weights)

        if weights.sum() == 0:
            return 0

        p = weights / weights.sum()

        entropy = -np.sum(p * np.log(p + 1e-10))

        return entropy

    # ----------------------------------
    # 6. Claim-Level Uncertainty
    # ----------------------------------
    def claim_uncertainty(self, G):

        scores = {}

        for node in G.nodes():

            outgoing = sum([G[node][nbr]['weight'] for nbr in G.successors(node)])
            incoming = sum([G[nbr][node]['weight'] for nbr in G.predecessors(node)])

            total = outgoing + incoming

            scores[node] = 1 / (total + 1e-6)

        return scores

    # ----------------------------------
    # 7. Response-Level Score
    # ----------------------------------
    def response_uncertainty(self, claim_scores, mapping):

        response_scores = {}

        for idx, resp_id in enumerate(mapping):
            response_scores.setdefault(resp_id, []).append(claim_scores[idx])

        # average uncertainty per response
        return {
            r: np.mean(scores)
            for r, scores in response_scores.items()
        }

    # ----------------------------------
    # 8. Full Pipeline
    # ----------------------------------
    def run(self, responses):

        claims, mapping = self.extract_claims(responses)

        G = self.build_graph(claims)

        entropy = self.structural_entropy(G)

        claim_scores = self.claim_uncertainty(G)

        response_scores = self.response_uncertainty(claim_scores, mapping)

        return {
            "claims": claims,
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
            "entropy": entropy,
            "claim_uncertainty": claim_scores,
            "response_uncertainty": response_scores
        }