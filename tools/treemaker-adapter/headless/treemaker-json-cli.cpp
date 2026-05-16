#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <memory>
#include <cmath>
#include <regex>
#include <stdexcept>
#include <vector>

#include "tmModel.h"
#include "tmNLCO.h"
#include "tmStubFinder.h"

using namespace std;

static string json_escape(const string& value) {
  string out;
  for (char c : value) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out;
}

static string kind_name(tmCrease::Kind kind) {
  switch (kind) {
    case tmCrease::AXIAL: return "AXIAL";
    case tmCrease::GUSSET: return "GUSSET";
    case tmCrease::RIDGE: return "RIDGE";
    case tmCrease::UNFOLDED_HINGE: return "UNFOLDED_HINGE";
    case tmCrease::FOLDED_HINGE: return "FOLDED_HINGE";
    case tmCrease::PSEUDOHINGE: return "PSEUDOHINGE";
  }
  return "UNKNOWN";
}

static string assignment_name(tmCrease::Fold fold, tmCrease::Kind kind) {
  if (fold == tmCrease::MOUNTAIN) return "M";
  if (fold == tmCrease::VALLEY) return "V";
  if (fold == tmCrease::BORDER) return "B";
  if (kind == tmCrease::UNFOLDED_HINGE) return "F";
  return "F";
}

static int fold_angle(tmCrease::Fold fold) {
  if (fold == tmCrease::MOUNTAIN) return -180;
  if (fold == tmCrease::VALLEY) return 180;
  return 0;
}

static const char* cp_status_name(tmTree::CPStatus status) {
  switch (status) {
    case tmTree::HAS_FULL_CP: return "HAS_FULL_CP";
    case tmTree::EDGES_TOO_SHORT: return "EDGES_TOO_SHORT";
    case tmTree::POLYS_NOT_VALID: return "POLYS_NOT_VALID";
    case tmTree::POLYS_NOT_FILLED: return "POLYS_NOT_FILLED";
    case tmTree::POLYS_MULTIPLE_IBPS: return "POLYS_MULTIPLE_IBPS";
    case tmTree::VERTICES_LACK_DEPTH: return "VERTICES_LACK_DEPTH";
    case tmTree::FACETS_NOT_VALID: return "FACETS_NOT_VALID";
    case tmTree::NOT_LOCAL_ROOT_CONNECTABLE: return "NOT_LOCAL_ROOT_CONNECTABLE";
  }
  return "UNKNOWN";
}


static tmTree* make_triangle_tree() {
  tmTree* tree = new tmTree();
  tree->SetPaperWidth(1.0);
  tree->SetPaperHeight(1.0);
  tree->SetScale(1.0);
  tmNode* root = nullptr;
  tmEdge* rootEdge = nullptr;
  tree->AddNode(nullptr, tmPoint(0.5, 0.5), root, rootEdge);
  tmNode* a = nullptr; tmEdge* ea = nullptr;
  tmNode* b = nullptr; tmEdge* eb = nullptr;
  tmNode* c = nullptr; tmEdge* ec = nullptr;
  tree->AddNode(root, tmPoint(0.0, 0.0), a, ea);
  tree->AddNode(root, tmPoint(1.0, 0.0), b, eb);
  tree->AddNode(root, tmPoint(0.0, 1.0), c, ec);
  const tmFloat lA = (2.0 - std::sqrt(2.0)) / 2.0;
  const tmFloat lB = std::sqrt(2.0) / 2.0;
  const tmFloat lC = std::sqrt(2.0) / 2.0;
  ea->SetLength(lA);
  eb->SetLength(lB);
  ec->SetLength(lC);
  return tree;
}

struct JsonNode {
  string id;
  double x;
  double y;
};

struct JsonEdge {
  string from;
  string to;
  double length;
};

static string read_text_file(const string& path) {
  ifstream in(path.c_str());
  if (!in.good()) throw runtime_error("cannot open input");
  stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

static string array_section(const string& json, const string& key) {
  const string needle = "\"" + key + "\"";
  const size_t keyPos = json.find(needle);
  if (keyPos == string::npos) throw runtime_error("missing array: " + key);
  const size_t start = json.find('[', keyPos);
  if (start == string::npos) throw runtime_error("missing array open: " + key);
  int depth = 0;
  for (size_t i = start; i < json.size(); ++i) {
    if (json[i] == '[') depth++;
    else if (json[i] == ']') {
      depth--;
      if (depth == 0) return json.substr(start + 1, i - start - 1);
    }
  }
  throw runtime_error("missing array close: " + key);
}

static vector<string> object_blocks(const string& section) {
  vector<string> objects;
  int depth = 0;
  size_t start = string::npos;
  for (size_t i = 0; i < section.size(); ++i) {
    if (section[i] == '{') {
      if (depth == 0) start = i;
      depth++;
    } else if (section[i] == '}') {
      depth--;
      if (depth == 0 && start != string::npos) {
        objects.push_back(section.substr(start, i - start + 1));
        start = string::npos;
      }
    }
  }
  return objects;
}

static string json_string_field(const string& object, const string& key) {
  regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  smatch match;
  if (!regex_search(object, match, pattern)) throw runtime_error("missing string field: " + key);
  return match[1].str();
}

static double json_number_field(const string& object, const string& key) {
  regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
  smatch match;
  if (!regex_search(object, match, pattern)) throw runtime_error("missing number field: " + key);
  return stod(match[1].str());
}

static tmTree* make_tree_from_spec_json(const string& json) {
  vector<JsonNode> nodes;
  vector<JsonEdge> edges;
  for (const string& object : object_blocks(array_section(json, "nodes"))) {
    nodes.push_back(JsonNode{
      json_string_field(object, "id"),
      json_number_field(object, "x"),
      json_number_field(object, "y"),
    });
  }
  for (const string& object : object_blocks(array_section(json, "edges"))) {
    edges.push_back(JsonEdge{
      json_string_field(object, "from"),
      json_string_field(object, "to"),
      json_number_field(object, "length"),
    });
  }
  if (nodes.empty()) throw runtime_error("spec has no nodes");

  map<string, JsonNode> nodeById;
  for (const JsonNode& node : nodes) nodeById[node.id] = node;
  const string rootId = nodeById.count("root") ? "root" : nodes[0].id;

  tmTree* tree = new tmTree();
  tmNode* rootNode = nullptr;
  tmEdge* rootEdge = nullptr;
  const JsonNode& root = nodeById[rootId];
  tree->AddNode(nullptr, tmPoint(root.x, root.y), rootNode, rootEdge);

  map<string, tmNode*> made;
  made[rootId] = rootNode;
  vector<bool> used(edges.size(), false);
  size_t progress = 1;
  while (made.size() < nodes.size() && progress > 0) {
    progress = 0;
    for (size_t i = 0; i < edges.size(); ++i) {
      if (used[i]) continue;
      const JsonEdge& edge = edges[i];
      string parentId;
      string childId;
      if (made.count(edge.from) && !made.count(edge.to)) {
        parentId = edge.from;
        childId = edge.to;
      } else if (made.count(edge.to) && !made.count(edge.from)) {
        parentId = edge.to;
        childId = edge.from;
      } else {
        continue;
      }
      const JsonNode& child = nodeById[childId];
      tmNode* childNode = nullptr;
      tmEdge* childEdge = nullptr;
      tree->AddNode(made[parentId], tmPoint(child.x, child.y), childNode, childEdge);
      if (childEdge) childEdge->SetLength(edge.length);
      made[childId] = childNode;
      used[i] = true;
      progress++;
    }
  }
  if (made.size() != nodes.size()) {
    delete tree;
    throw runtime_error("spec graph is disconnected or cyclic in a way the adapter cannot build");
  }
  tree->SetScale(0.1);
  return tree;
}

static void usage() {
  cerr << "Usage: treemaker-json-cli (--spec spec.json | --in input.tmd5) --out output.json [--no-optimize] [--activate-leaf-paths] [--triangulate]" << endl;
}

int main(int argc, char** argv) {
  string input;
  string specPath;
  string output;
  bool optimize = true;
  bool activateLeafPaths = false;
  bool triangulate = false;
  for (int i = 1; i < argc; ++i) {
    string arg(argv[i]);
    if (arg == "--in" && i + 1 < argc) input = argv[++i];
    else if (arg == "--spec" && i + 1 < argc) specPath = argv[++i];
    else if (arg == "--out" && i + 1 < argc) output = argv[++i];
    else if (arg == "--no-optimize") optimize = false;
    else if (arg == "--activate-leaf-paths") activateLeafPaths = true;
    else if (arg == "--triangulate") triangulate = true;
    else if (arg == "--help" || arg == "-h") { usage(); return 0; }
    else { cerr << "Unknown argument: " << arg << endl; usage(); return 2; }
  }
  if ((input.empty() && specPath.empty()) || output.empty()) { usage(); return 2; }

  cout.setf(ios_base::fixed);
  cout.precision(6);
  tmPart::InitTypes();

  unique_ptr<tmTree> ownedTree;
  tmTree* treePtr = nullptr;
  tmTree stackTree;
  try {
    if (!specPath.empty()) {
      ownedTree.reset(make_tree_from_spec_json(read_text_file(specPath)));
      treePtr = ownedTree.get();
    } else if (input == "@optimized") {
      ownedTree.reset(tmTree::MakeTreeOptimized());
      treePtr = ownedTree.get();
    } else if (input == "@gusset") {
      ownedTree.reset(tmTree::MakeTreeGusset());
      treePtr = ownedTree.get();
    } else if (input == "@conditioned") {
      ownedTree.reset(tmTree::MakeTreeConditioned());
      treePtr = ownedTree.get();
    } else if (input == "@triangle") {
      ownedTree.reset(make_triangle_tree());
      treePtr = ownedTree.get();
    } else {
      ifstream fin(input.c_str());
      if (!fin.good()) throw runtime_error("cannot open input");
      stackTree.GetSelf(fin);
      treePtr = &stackTree;
    }
  } catch (const exception& e) {
    cerr << "TreeMaker read failed: " << e.what() << endl;
    return 3;
  } catch (...) {
    cerr << "TreeMaker read failed" << endl;
    return 3;
  }

  if (activateLeafPaths) {
    tmArray<tmPath*> allLeafPaths;
    treePtr->GetLeafPaths(allLeafPaths);
    treePtr->SetPathsActive(allLeafPaths);
  }

  bool optimizationSuccess = true;
  string optimizationError;
  if (optimize) {
    tmNLCO_alm nlco;
    tmScaleOptimizer optimizer(treePtr, &nlco);
    try {
      optimizer.Initialize();
      optimizer.Optimize();
    } catch (tmNLCO::EX_BAD_CONVERGENCE ex) {
      optimizationSuccess = false;
      optimizationError = string("bad convergence: ") + to_string(ex.GetReason());
    } catch (tmScaleOptimizer::EX_BAD_SCALE&) {
      optimizationSuccess = false;
      optimizationError = "bad scale";
    } catch (const exception& e) {
      optimizationSuccess = false;
      optimizationError = e.what();
    } catch (...) {
      optimizationSuccess = false;
      optimizationError = "unknown optimization failure";
    }
  }

  bool buildSuccess = true;
  string buildError;
  try {
    if (triangulate) {
      tmStubFinder stubFinder(treePtr);
      stubFinder.TriangulateTree();
    }
    treePtr->BuildPolysAndCreasePattern();
  } catch (const exception& e) {
    buildSuccess = false;
    buildError = e.what();
  } catch (...) {
    buildSuccess = false;
    buildError = "unknown build failure";
  }

  tmTree::CPStatus cpStatus = treePtr->HasFullCP() ? tmTree::HAS_FULL_CP : tmTree::POLYS_NOT_VALID;

  ofstream out(output.c_str());
  if (!out.good()) { cerr << "cannot open output" << endl; return 4; }
  out.setf(ios_base::fixed);
  out.precision(10);
  out << "{\n";
  out << "  \"schemaVersion\": \"treemaker-output/v1\",\n";
  out << "  \"generator\": \"TreeMaker model CLI\",\n";
  out << "  \"toolVersion\": \"treemaker-legacy-headless/v0.1.0\",\n";
  out << "  \"ok\": " << ((optimizationSuccess && buildSuccess && treePtr->HasFullCP()) ? "true" : "false") << ",\n";
  out << "  \"optimization\": {\"success\": " << (optimizationSuccess ? "true" : "false") << ", \"error\": \"" << json_escape(optimizationError) << "\"},\n";
  out << "  \"foldedForm\": {\"success\": " << (treePtr->HasFullCP() ? "true" : "false") << ", \"cpStatus\": \"" << cp_status_name(cpStatus) << "\"},\n";
  out << "  \"stats\": {\"vertices\": " << treePtr->GetVertices().size() << ", \"creases\": " << treePtr->GetCreases().size() << ", \"facets\": " << treePtr->GetFacets().size() << "},\n";
  out << "  \"creases\": [\n";
  const tmDpptrArray<tmCrease>& creases = treePtr->GetCreases();
  for (size_t i = 0; i < creases.size(); ++i) {
    const tmCrease* crease = creases[i];
    const tmDpptrArray<tmVertex>& vertices = crease->GetVertices();
    const tmPoint& p1 = vertices[0]->GetLoc();
    const tmPoint& p2 = vertices[1]->GetLoc();
    string assignment = assignment_name(crease->GetFold(), crease->GetKind());
    out << "    {\"p1\": [" << p1.x << ", " << p1.y << "], \"p2\": [" << p2.x << ", " << p2.y << "], ";
    out << "\"assignment\": \"" << assignment << "\", \"foldAngle\": " << fold_angle(crease->GetFold()) << ", ";
    out << "\"kind\": \"" << kind_name(crease->GetKind()) << "\"}";
    if (i + 1 < creases.size()) out << ",";
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  return 0;
}
