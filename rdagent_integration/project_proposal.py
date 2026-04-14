"""Hypothesis2Experiment classes that build experiments with project-local Qlib YAML templates."""

from __future__ import annotations

import json

from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from rdagent.components.coder.model_coder.model import ModelExperiment, ModelTask
from rdagent.core.proposal import Hypothesis, Trace
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.proposal.factor_proposal import QlibFactorHypothesis2Experiment
from rdagent.scenarios.qlib.proposal.model_proposal import QlibModelHypothesis2Experiment

from rdagent_integration.project_experiments import ProjectQlibFactorExperiment, ProjectQlibModelExperiment


class ProjectQlibFactorHypothesis2Experiment(QlibFactorHypothesis2Experiment):
    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        response_dict = json.loads(response)
        tasks = []

        for factor_name in response_dict:
            description = response_dict[factor_name]["description"]
            formulation = response_dict[factor_name]["formulation"]
            variables = response_dict[factor_name]["variables"]
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    variables=variables,
                )
            )

        exp = ProjectQlibFactorExperiment(tasks, hypothesis=hypothesis)
        exp.based_experiments = [ProjectQlibFactorExperiment(sub_tasks=[])] + [
            t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
        ]

        unique_tasks = []
        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                if isinstance(based_exp, QlibModelExperiment):
                    continue
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.sub_tasks = unique_tasks
        return exp


class ProjectQlibModelHypothesis2Experiment(QlibModelHypothesis2Experiment):
    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> ModelExperiment:
        response_dict = json.loads(response)
        tasks = []
        for model_name in response_dict:
            description = response_dict[model_name]["description"]
            formulation = response_dict[model_name]["formulation"]
            architecture = response_dict[model_name]["architecture"]
            variables = response_dict[model_name]["variables"]
            hyperparameters = response_dict[model_name]["hyperparameters"]
            training_hyperparameters = response_dict[model_name]["training_hyperparameters"]
            model_type = response_dict[model_name]["model_type"]
            tasks.append(
                ModelTask(
                    name=model_name,
                    description=description,
                    formulation=formulation,
                    architecture=architecture,
                    variables=variables,
                    hyperparameters=hyperparameters,
                    training_hyperparameters=training_hyperparameters,
                    model_type=model_type,
                )
            )
        exp = ProjectQlibModelExperiment(tasks, hypothesis=hypothesis)
        exp.based_experiments = [t[0] for t in trace.hist if t[1] and isinstance(t[0], ModelExperiment)]
        return exp
