import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import time

st.set_page_config(
    page_title="Campus Shuttle Optimizer",
    page_icon="🚌",
    layout="wide"
)

# ---------------- CAMPUS DATA ----------------

locations = {
    "A": "Main Gate",
    "B": "Hostel",
    "C": "Academic Block",
    "D": "Library",
    "E": "Cafeteria",
    "F": "Sports Complex",
    "G": "Admin Block",
    "H": "Auditorium",
    "I": "Medical Centre",
    "J": "Parking Area"
}

roads = [
    ("A", "B", 4), ("A", "G", 3),
    ("B", "C", 5), ("B", "I", 4),
    ("C", "D", 2), ("C", "G", 3),
    ("D", "E", 3), ("D", "H", 4),
    ("E", "F", 5), ("F", "J", 3),
    ("G", "H", 4), ("H", "I", 3),
    ("I", "J", 5)
]

graph = {node: [] for node in locations}

for u, v, weight in roads:
    graph[u].append((v, weight))
    graph[v].append((u, weight))

# ---------------- ALGORITHMS ----------------

def dijkstra(graph, start):
    dist = {u: float("inf") for u in graph}
    prev = {u: None for u in graph}
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        for v, weight in graph[u]:
            new_dist = current_dist + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, prev


def get_path(previous, start, end):
    if start == end:
        return [start]

    if previous[end] is None:
        return []

    path = []
    current = end

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()
    return path if path and path[0] == start else []


def floyd_warshall(graph):
    nodes = list(graph)
    dist = {u: {v: float("inf") for v in nodes} for u in nodes}

    for u in nodes:
        dist[u][u] = 0
        for v, weight in graph[u]:
            dist[u][v] = weight

    for k in nodes:
        for i in nodes:
            for j in nodes:
                dist[i][j] = min(
                    dist[i][j],
                    dist[i][k] + dist[k][j]
                )

    return dist


def prim_mst():
    start = "A"
    visited = {start}
    edges = []
    pq = []

    for v, weight in graph[start]:
        heapq.heappush(pq, (weight, start, v))

    while pq and len(visited) < len(graph):
        weight, u, v = heapq.heappop(pq)

        if v in visited:
            continue

        visited.add(v)
        edges.append((u, v, weight))

        for neighbor, w in graph[v]:
            if neighbor not in visited:
                heapq.heappush(pq, (w, v, neighbor))

    return edges


def kruskal_mst():
    parent = {u: u for u in graph}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    result = []

    for u, v, weight in sorted(roads, key=lambda x: x[2]):
        root_u, root_v = find(u), find(v)

        if root_u != root_v:
            parent[root_u] = root_v
            result.append((u, v, weight))

        if len(result) == len(graph) - 1:
            break

    return result


# ---------------- DASHBOARD ----------------

st.title("🚌 Campus Shuttle Routing & Capacity Scheduling")
st.caption("DAA Project | Interactive demonstration with synthetic data")
st.info("The locations, road times and passenger requests are illustrative, not measured campus data.")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Campus Network",
        "Shortest Route Finder",
        "Shuttle Scheduling",
        "Capacity Optimization",
        "Algorithm Comparison"
    ]
)

if page == "Home":
    st.header("Campus Shuttle Optimizer")

    c1, c2, c3 = st.columns(3)
    c1.metric("Campus Locations", len(locations))
    c2.metric("Road Connections", len(roads))
    c3.metric("Algorithms", 6)

    st.write("""
    This dashboard demonstrates graph algorithms for campus network
    connectivity, shortest routes, shuttle request prioritization,
    capacity optimization and performance analysis.
    """)

elif page == "Campus Network":
    st.header("Campus Network")

    G = nx.Graph()
    for u, name in locations.items():
        G.add_node(u, label=name)

    for u, v, w in roads:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10, 6))

    nx.draw(
        G, pos, ax=ax, with_labels=True,
        node_color="skyblue", node_size=1000,
        font_weight="bold", edge_color="gray"
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=nx.get_edge_attributes(G, "weight"),
        ax=ax
    )

    labels = {u: f"{u}: {name}" for u, name in locations.items()}
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=7, ax=ax,
        verticalalignment="bottom"
    )

    ax.set_title("Campus Locations and Road Travel Times (minutes)")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Road Data")
    st.dataframe(
        [{"From": locations[u], "To": locations[v],
          "Travel Time (min)": w} for u, v, w in roads],
        use_container_width=True
    )

elif page == "Shortest Route Finder":
    st.header("Shortest Route Finder")

    source = st.selectbox(
        "Starting location", list(locations),
        format_func=lambda x: locations[x]
    )
    destination = st.selectbox(
        "Destination", list(locations), index=3,
        format_func=lambda x: locations[x]
    )

    if st.button("Find Shortest Route"):
        distances, previous = dijkstra(graph, source)
        path = get_path(previous, source, destination)

        if path:
            st.success(f"Minimum travel time: {distances[destination]} minutes")
            st.write(" → ".join(locations[x] for x in path))

            G = nx.Graph()
            G.add_weighted_edges_from(roads)
            pos = nx.spring_layout(G, seed=42)
            fig, ax = plt.subplots(figsize=(9, 5))

            nx.draw(
                G, pos, ax=ax, with_labels=True,
                node_color="lightblue", edge_color="lightgray",
                node_size=800
            )
            nx.draw_networkx_edges(
                G, pos, edgelist=list(zip(path, path[1:])),
                edge_color="red", width=4, ax=ax
            )
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=nx.get_edge_attributes(G, "weight"),
                ax=ax
            )

            st.pyplot(fig)
            plt.close(fig)
        else:
            st.error("No route exists.")

elif page == "Shuttle Scheduling":
    st.header("Greedy Shuttle Scheduling")
    st.write("Priority score = passenger count × waiting time")

    requests = [
        {"Stop": "Hostel", "Passengers": 12, "Waiting": 10},
        {"Stop": "Library", "Passengers": 5, "Waiting": 20},
        {"Stop": "Cafeteria", "Passengers": 15, "Waiting": 5},
        {"Stop": "Sports Complex", "Passengers": 8, "Waiting": 15},
        {"Stop": "Academic Block", "Passengers": 10, "Waiting": 12}
    ]

    for request in requests:
        request["Priority"] = request["Passengers"] * request["Waiting"]

    requests.sort(key=lambda x: x["Priority"], reverse=True)
    st.dataframe(requests, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [r["Stop"] for r in requests],
        [r["Priority"] for r in requests]
    )
    ax.set_xlabel("Priority Score")
    ax.set_title("Greedy Shuttle Request Priorities")
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close(fig)

elif page == "Capacity Optimization":
    st.header("Dynamic Programming: Capacity Optimization")

    passenger_counts = [12, 8, 10, 5, 15]
    waiting_scores = [10, 15, 12, 20, 5]
    stops = ["Hostel", "Sports Complex", "Academic Block",
             "Library", "Cafeteria"]

    capacity = st.slider("Shuttle capacity", 5, 40, 25)
    n = len(stops)

    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]

            if passenger_counts[i - 1] <= w:
                dp[i][w] = max(
                    dp[i][w],
                    waiting_scores[i - 1]
                    + dp[i - 1][w - passenger_counts[i - 1]]
                )

    selected = []
    w = capacity

    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= passenger_counts[i - 1]

    selected.reverse()
    total_passengers = sum(passenger_counts[i] for i in selected)

    c1, c2, c3 = st.columns(3)
    c1.metric("Capacity", capacity)
    c2.metric("Passengers Selected", total_passengers)
    c3.metric("Maximum Score", dp[n][capacity])

    st.write("Selected stops:")
    if selected:
        st.dataframe([
            {
                "Stop": stops[i],
                "Passengers": passenger_counts[i],
                "Waiting Score": waiting_scores[i]
            } for i in selected
        ], use_container_width=True)
    else:
        st.write("No passenger groups selected.")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [
        "green" if i in selected else "lightgray"
        for i in range(n)
    ]
    ax.bar(stops, passenger_counts, color=colors)
    ax.set_ylabel("Passengers")
    ax.set_title("Selected Passenger Groups (Green)")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)
    plt.close(fig)

elif page == "Algorithm Comparison":
    st.header("Algorithm Comparison")

    st.subheader("Minimum Spanning Tree")
    prim_edges = prim_mst()
    kruskal_edges = kruskal_mst()

    c1, c2 = st.columns(2)
    c1.metric("Prim MST Total", sum(w for _, _, w in prim_edges))
    c2.metric("Kruskal MST Total", sum(w for _, _, w in kruskal_edges))

    st.write("Prim's selected edges")
    st.dataframe([
        {"From": locations[u], "To": locations[v], "Time": w}
        for u, v, w in prim_edges
    ], use_container_width=True)

    st.write("Kruskal's selected edges")
    st.dataframe([
        {"From": locations[u], "To": locations[v], "Time": w}
        for u, v, w in kruskal_edges
    ], use_container_width=True)

    st.subheader("Shortest-Path Algorithms")
    source = st.selectbox(
        "Source for comparison", list(locations),
        format_func=lambda x: locations[x],
        key="comparison_source"
    )

    start = time.perf_counter()
    dijkstra(graph, source)
    dijkstra_time = time.perf_counter() - start

    start = time.perf_counter()
    floyd_warshall(graph)
    floyd_time = time.perf_counter() - start

    st.write(f"Dijkstra runtime: {dijkstra_time:.8f} seconds")
    st.write(f"Floyd–Warshall runtime: {floyd_time:.8f} seconds")

    st.caption("Runtime values vary by run and machine. This small synthetic graph is for demonstration.")
