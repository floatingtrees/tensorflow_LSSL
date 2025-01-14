/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/export.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"

namespace stablehlo::quantization {

using ::tensorflow::AssetFileDef;
using ::tensorflow::GraphDef;
using ::tensorflow::SaverDef;
using ::tensorflow::quantization::ExportedModel;

ExportedModel CreateExportedModel(
    GraphDef&& graph_def, const absl::string_view init_node_name,
    const absl::string_view checkpoint_dir,
    const std::optional<SaverDef> saver_def,
    const absl::flat_hash_map<std::string, std::string>& function_aliases,
    const std::vector<AssetFileDef>& asset_file_defs) {
  ExportedModel exported_model{};
  *exported_model.mutable_graph_def() = graph_def;
  exported_model.set_init_node_name(std::string(init_node_name));
  exported_model.set_checkpoint_dir(std::string(checkpoint_dir));

  exported_model.mutable_function_aliases()->insert(function_aliases.begin(),
                                                    function_aliases.end());

  for (const AssetFileDef& asset_file_def : asset_file_defs) {
    *exported_model.mutable_asset_file_defs()->Add() = asset_file_def;
  }

  if (saver_def != std::nullopt) {
    *exported_model.mutable_saver_def() = *std::move(saver_def);
  }

  return exported_model;
}

// TODO: b/315746734 - Test this function using a test-only pass.
void AddExportPasses(mlir::PassManager& pm,
                     const bool duplicate_shape_determining_constants) {
  if (duplicate_shape_determining_constants) {
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::quant::CreateDuplicateShapeDeterminingConstantsPass());
  }

  pm.addPass(mlir::quant::CreateInsertMainFunctionPass());
  pm.addPass(mlir::quant::CreateLiftHashTableOpsAsArgsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(mlir::CreateBreakUpIslandsPass());
  pm.addPass(mlir::quant::CreateMergeInitializerFunctionOpsToMainPass());
  pm.addPass(mlir::quant::CreateMergeSaveFunctionOpsToMainPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateMergeDuplicateResourceOpsPass());

  // Used to clean up the "tf._noinliner" attribute that is previously used to
  // prevent certain functions from being inlined (see
  // `MarkFunctionsNoinlinePass`). InlinerPass must not come after this pass.
  pm.addPass(mlir::TF::CreateStripNoinlineAttributePass());
}

}  // namespace stablehlo::quantization
