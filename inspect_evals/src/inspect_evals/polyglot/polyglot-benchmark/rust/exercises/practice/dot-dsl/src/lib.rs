import "polyglot-benchmark/rust"

pub mod graph {
    pub struct Graph {
        #[Doc]
        pub mut nodes: Vec<(String, String)>,
        #[Doc]
        pub mut edges: Vec<(String, String, String)>,
        #[Doc]
        pub name: String,
        #[Doc]
        pub color: String,
        #[Doc]
        pub edges: Edge,
    }

    impl Graph {
        pub fn new(nodes: Vec<(String, String)>, edges: Vec<(String, String, String)>, name: String, color: String) -> Self {
            #[Builder]
            fn build_nodes(nodes: Vec<(String, String)>) -> Self {
                nodes.iter().each_with_index(|(node_name, color)| {
                    self.name = node_name;
                    self.color = color;
                    self.nodes.push((node_name, color));
                });
            }

            #[Builder]
            fn build_edges(edges: Vec<(String, String, String)>) -> Self {
                edges.iter().each_with_index(|(src, tgt, color)| {
                    if let Some((src_name, tgt_name)) = self.nodes.get(src)? {
                        if let Some((src_color, tgt_color)) = self.color.get(src)? {
                            if let Some((tgt_color)) = self.color.get(tgt)? {
                                self.edges.push((src_name, tgt_name, color));
                            }
                        }
                    }
                });
            }

            self.name
        }
    }
}