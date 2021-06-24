#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GeneticML (Genetic Algorithm)

Created on May 2020
@author: Joaquin Rives
"""

import numpy as np
import pandas as pd
import copy
import random
from indexed import IndexedOrderedDict
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from itertools import chain
import uuid
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
from multiprocessing.pool import ThreadPool
import warnings

class Gene:

    allowed_encodings = ['real', 'binary', 'gray', 'categorical', 'binaryInt', 'grayInt', 'realInt']

    def __init__(self, name, encoding, minv=None, maxv=None, length=None, precision=None,
                 categorical_values=None, mutation_prob=None, cooperates=None, coop_relationship=None):

        if encoding not in self.allowed_encodings:
            raise AttributeError(f"Encoding '{encoding}' is not allowed (allowed encodings: {self.allowed_encodings}.")

        if cooperates and not coop_relationship:
            raise AttributeError("If a gene cooperates with one or more other genes for the expression of the "
                                 "phenotype the relationship must be specified.")

        if encoding != 'categorical' and (minv is None or maxv is None):
            raise AttributeError("The maximum and minimum values for a non-categorical gene must be provided.")

        if precision and length:
            warnings.warn(f"Warning: Only one (length or precision) can be specified. The last one will be estimated.")

        if encoding == 'categorical' and not (categorical_values or isinstance(categorical_values, (list, tuple))):
            raise AttributeError("For a gene of type 'categorical' the attribute 'values_dict' must contain a "
                                 "dictionary with the different possible values of the gene.")

        self.name = name
        self.encoding = encoding
        self.minv = minv
        self.maxv = maxv
        self.length = 1 if (encoding in ['categorical', 'real', 'realInt']) else int(length) if length else None  # Todo refractor
        self.precision = precision
        self.mutation_prob = mutation_prob
        self.cooperates = cooperates  # TODO
        self.relationship = coop_relationship

        if self.encoding in ['binary', 'gray', 'grayInt', 'binaryInt']:
            if self.length:
                self.precision = self.calc_precision(self.minv, self.maxv, self.length)
            elif self.precision:
                self.length = self.calc_length_binary(self.minv, self.maxv, self.precision)
            else:
                raise AttributeError(f"Neither 'length' or 'precision' of the gene was provided. At least one of them "
                                f"must be specified for {self.encoding} encoding.")

        # gene values
        self.categorical_values = categorical_values
        self._genotype = None
        self.phenotype = None
        if self.encoding == 'categorical':
            self.categorical_dict = {i: value for i, value in enumerate(self.categorical_values)}

        # chromosome
        self.chromosome = None
        self.location = None

    @property
    def genotype(self):
        return self._genotype

    @genotype.setter
    def genotype(self, value):
        """ Whenever the genotype is set or changed _express() is triggered and the phenotype is updated. Also the
        chromosome sequence gets updated."""
        self._genotype = value
        self._express()
        if self.chromosome:
            if self.chromosome.individual:
                self.chromosome.individual.phenotype.update({self.name: self.phenotype})
            if not np.array_equal(self.chromosome.sequence[self.location[0]: self.location[1]], self.genotype):
                self.chromosome.sequence[self.location[0]: self.location[1]] = self.genotype

    def _express(self):
        if self.encoding == 'categorical':
            self.phenotype = self.categorical_dict[int(self.genotype)]
        elif self.encoding == 'real':
            self.phenotype = self.genotype
        elif self.encoding == 'realInt':
            self.phenotype = self.to_int()
        elif self.encoding in ['binary', 'gray']:
            self.phenotype = self.to_real()
        elif self.encoding in ['binaryInt', 'grayInt']:
            self.phenotype = self.to_int()
        else:
            warnings.warn(f"The Gene {self.name} with genotype {self.genotype} could not be expressed.")

    def to_int(self):
        return int(self.to_real())

    def to_binary(self):
        if self.encoding == 'categorical':
            warnings.warn("Gene of type categorical can not be converted to binary.")
            return None

        elif self.encoding in ['binary', 'binaryInt']:
            return self.genotype

        elif self.encoding in ['gray', 'grayInt']:
            return self.gray2binary(self.genotype)

        elif self.encoding in ['real', 'realInt']:
            return np.array([int(bit) for bit in np.binary_repr(self.genotype)])
        else:
            warnings.warn("Gene couldn't be converted to binary.")
            return None

    def to_gray(self):
        if self.encoding == 'categorical':
            warnings.warn("Gene of type categorical can not be converted to binary.")
            return None

        elif self.encoding == 'gray':
            return self.genotype

        elif self.encoding == 'binary':
            binary = list(self.genotype)
            gray = [binary[0]]
            for i in range(1, len(binary)):
                gray.append(0) if (binary[i - 1] == binary[i]) else gray.append(1)
            return np.array(gray)

        elif self.encoding in ['real', 'realInt']:
            binary = [int(bit) for bit in np.binary_repr(self.genotype)]
            gray = [binary[0]]
            for i in range(1, len(binary)):
                gray.append(0) if (binary[i - 1] == binary[i]) else gray.append(1)
            return np.array(gray)

        else:
            warnings.warn("Gene couldn't be converted to gray.")
            return None

    def to_real(self):
        if self.encoding == 'categorical':
            warnings.warn("Gene couldn't be converted to gray.")
            return None

        elif self.encoding in ['real', 'realInt']:
            return self.genotype

        elif self.encoding in ['binary', 'binaryInt']:
            z = 0
            val = 0
            for i in range(self.length - 1, -1, -1):
                vi = self.genotype[i] * 2 ** z
                val += vi
                z = z + 1
            value = (self.precision * val) + self.minv
            return value

        elif self.encoding in ['gray', 'grayInt']:

            binary = self.gray2binary(self.genotype)
            z = 0
            val = 0
            for i in range(self.length - 1, -1, -1):
                vi = binary[i] * 2 ** z
                val += vi
                z = z + 1
            value = (self.precision * val) + self.minv
            return value
        else:
            warnings.warn("Gene couldn't be converted to real.")
            return None

    @staticmethod
    def calc_precision(minv, maxv, length):
        return (maxv - minv) / (2 ** length - 1)

    @staticmethod
    def calc_length_binary(minv, maxv, precision):
        return int(np.ceil(np.log2(((maxv - minv) / precision) - 1)))

    @staticmethod
    def gray2binary(gray):
        binary = []
        gray = list(gray)
        binary.append(gray[0])
        for i in range(1, len(gray)):
            if gray[i] == 0:
                binary.append(binary[i - 1])
            else:
                binary.append(1) if (binary[i - 1] == 0) else binary.append(0)
        return np.array(binary)


class Chromosome:

    # Explain Poisson: If recombination='poisson' the number of recombination points per recombination follow a
    # poisson's distribution of mean equal to lambda. The default is lambda=1 (1 recombination point on average
    # per recombination).

    # r_prob: ... .If recombination='poisson' r_prob is ignored.

    allowed_recombination = ['single', 'double', 'poisson', False, None]
    allowed_mutations = ['uniform', 'gene_specific', False, None]

    def __init__(self, genes, name=None, mutation='uniform', recombination='single', r_prob=0.75, m_prob=0.15, r_lambda=1):

        if mutation not in self.allowed_mutations:
            raise AttributeError(f"Mutation {mutation} is not allowed (allowed mutations: {self.allowed_mutations}.")
        if recombination not in self.allowed_recombination:
            raise AttributeError(f"Recombination {recombination} is not allowed (allowed recombination: "
                             f"{self.allowed_recombination}.")

        if not isinstance(genes, (tuple, list)):
            genes = [genes]

        genes = copy.deepcopy(genes)

        self.name = name if name else str(uuid.uuid4())
        self.mutation = mutation
        self.m_prob = None if self.mutation == 'gene_specific' else m_prob

        loc_index = 0
        loc_end = 0
        for gene in genes:

            if self.mutation == 'gene_specific' and not gene.m_prob:
                raise ValueError(
                    f"The gene mutation_prob of the gene '{gene.name}' is 'None'. If mutation "
                    f"is set to 'gene_specific' each gene must contain its probability of mutation "
                    f"specified on its attributes.")

            if mutation == 'uniform':
                gene.m_prob = self.m_prob

            gene.chromosome = self  # backwards reference

            loc_start = loc_index
            loc_end = loc_start + gene.length
            loc_index = loc_end
            gene.location = (loc_start, loc_end)

        self.length = loc_end
        # self.genes = Prodict.from_dict({gene.name: gene for gene in genes})
        self.genes = IndexedOrderedDict({gene.name: gene for gene in genes})
        self.individual = None

        self._sequence = np.full(self.length, np.nan)

        # recombination
        self.recombination = recombination
        self.r_prob = r_prob
        self.r_lambda = r_lambda  # poisson's lambda

    @property
    def sequence(self):
        return self._sequence

    @sequence.setter
    def sequence(self, new_sequence):
        """ Whenever the chr sequence is set or changed update_genes() is triggered and the genes get updated.

        Warning: If the sequence is modifyed using slicing (e.g chr.sequence[:10] = 'foo') the setter won't
        get called and you should update the genetic information manually by calling 'chr.update_genes()'.
        """

        if not isinstance(new_sequence, np.ndarray):
            raise TypeError("The chr sequence has to be a np.ndarray.")
        if not self._sequence.shape == new_sequence.shape:
            raise ValueError('The new chr sequence must have the same length.')

        self._sequence = new_sequence
        self.update_genes()

    def update_genes(self):
        for gene in self.genes.values():
            if not np.array_equal(gene.genotype, self.sequence[gene.location[0]: gene.location[1]]):
                gene.genotype = self.sequence[gene.location[0]: gene.location[1]]

    def mutate(self):
        if self.mutation:
            temp_seq = copy.deepcopy(self.sequence)

            index = 0
            for gene in self.genes.values():
                for i in range(gene.length):
                    m_chance = np.random.rand()  # mutation chance
                    m_prob = gene.m_prob if self.mutation == 'gene_specific' else self.m_prob

                    if m_chance < m_prob:
                        if gene.encoding in ['real', 'realInt']:
                            temp_seq[index] = np.random.uniform(low=gene.minv, high=gene.maxv)
                        if gene.encoding == 'categorical':
                            temp_seq[index] = np.random.choice(list(gene.categorical_dict.keys() - gene.genotype))
                        if gene.encoding in ['binary', 'gray', 'binaryInt', 'grayInt']:
                            temp_seq[index] = 1 - temp_seq[index]
                    index += 1
            self.sequence = temp_seq

    def show(self):
        pass


class Individual:
    """
    When an instance of this class is created without specifying any parents the genome will be initialized randomly.
    To specify the parents when creating the instances use
    'Individual.from_parents(p1, p2=None, n_child=1)', the genome of the new created individual will be
    the result of the crossover of the parents plus mutations. The second parent is optional, if only one parent is
    provided there will be no crossover, but mutations can still happened.
    If the n_child is a odd number, the last remaining child will be discarded.

    info / tags: for user customization (e.g heritable_info / history / family_tree...), the will be passed to new
                 generation, in the from_parents() method you can easily modify the code and design how or what
                 information is inherited by the sons.

    Individual(chr_inheritance='independent'/'dependent'):
          - 'independent': the chromosomes are independent and the children have a 50% chance of inherit each of the
                         chromosomes from either of the parents.
          - 'dependent': the chromosomes of each parent won't get mixed and will be passed together to
                         the children. Recombination still may happens.

    init_genome(): Random initialization of the chromosomal sequences. When using the class method
                 Individual.from_parents() to create an instance, init_genome will be automatically set to False,
                 as it will be the genetic information of the parents which will be used to create the genetic
                 sequence of the new instance.
                 If you want to keep only some parts or genes of the sequence you can set init_genome=True and
                 set o modify any part of the sequence or gene afterwards, some examples:

                 individual.genes['gene1'].genotype = np.array([1,1,0,1,1])
                 or
                 individual.genome['chr1'].sequence = np.array([1,1,0...])

                 The sequence can be access through different ways, it doesn't matter, the variables are
                 inter-connected so that when a change of genotype or in the sequence is detected
                 all the dependent variables related to that sequence, such as the phenotypes, should get automatically
                 updated. There are some exceptions, if the genetic sequence modifyed using slicing
                 (e.g chr.sequence[:10] = 'foo'), it will bypass the setter method without triggering the update
                 of the genes and phenotypes that depend on that part of the sequence. So, you have to update the
                 genetic information manually by calling 'chr.update_genes()'.

    - When creating a population, the individual passed as argument doesn't need be initialize previously or have a
        defined sequence, the individual will just be used as a template. The Population will create the first
        generation by making copies of that individual and calling the Individual.init_genome() method for each
        new copy.

    - scores (dict): Dictionary with all the results/scores of the cross-validation of the individual.
                    The dictionary is actually just the output of GridSearchCV.cv_results_ for that individual, and
                    the scores can be accessed the same way.
                    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    """

    allowed_chr_inheritance = ['independent', 'uniparental']

    def __init__(self, genome, generation=0, info=None, tags=None, chr_inheritance='independent'):

        if chr_inheritance not in self.allowed_chr_inheritance:
            raise ValueError(f"chr_inheritance '{chr_inheritance}' is not allowed (allowed chr_inheritance: "
                             f"{self.allowed_chr_inheritance}.")

        if not isinstance(genome, (tuple, list)):
            genome = [genome]

        genome = copy.deepcopy(genome)

        self.id = str(uuid.uuid4())  # every time self.init_genome() is called a new id will be assigned
        self.generation = generation
        self.info = info
        self.tags = tags
        self.genes = IndexedOrderedDict()
        self.genome = IndexedOrderedDict({chrom.name: chrom for chrom in genome})
        self.phenotype = {}
        self.chr_inheritance = chr_inheritance
        self.scores = None
        self.fitness = None

        for chrom in self.genome.values():
            chrom.individual = self  # backwards reference

            for gene in chrom.genes.values():
                if gene.name in self.genes:
                    raise ValueError(f"There is already a chromosome in the genome that contains a gene with the name"
                                     f" '{gene.name}'.")
                else:
                    self.genes.update({gene.name: gene})

    @classmethod
    def from_parents(cls, p1, p2=None, n_children=1, chr_inheritance=None, info=None, tags=None):
        info_c, tags_c = copy.deepcopy(info), copy.deepcopy(tags)
        # info_c2, tags_c2 = copy.deepcopy(info), copy.deepcopy(tags)

        if not chr_inheritance:
            chr_inheritance = p1.chr_inheritance

        if chr_inheritance not in cls.allowed_chr_inheritance:
            raise ValueError(f"chr_inheritance '{chr_inheritance}' is not allowed (allowed chr_inheritance: "
                             f"{cls.allowed_chr_inheritance}.")

        generation = p1.generation + 1

        children = []
        while len(children) < n_children:
            genome_c1 = copy.deepcopy(p1.genome)
            genome_c2 = copy.deepcopy(p2.genome) if p2 else None

            # recombination
            if genome_c2:
                for chr_c1, chr_c2 in zip(genome_c1.keys(), genome_c2.keys()):

                    if genome_c1[chr_c1].recombination:
                        genome_c1[chr_c1], genome_c2[chr_c2] = cls.recombination(genome_c1[chr_c1], genome_c2[chr_c2])

                    if chr_inheritance == 'independent':
                        if np.random.rand() < 0.5:  # 50 % chance of inheriting the chr from either of the parents
                            temp_chr_c2 = copy.deepcopy(genome_c2[chr_c2])
                            genome_c2[chr_c2] = copy.deepcopy(genome_c1[chr_c1])
                            genome_c1[chr_c1] = temp_chr_c2
            # mutations
            for chrom in genome_c1.values():
                chrom.mutate()

            if genome_c2:
                for chrom in genome_c2.values():
                    chrom.mutate()

            children.append(cls(genome=list(genome_c1.values()), generation=generation, info=info_c,
                                tags=tags_c))
            if genome_c2:
                children.append(
                    cls(genome=list(genome_c2.values()), generation=generation, info=info_c,
                        tags=tags_c))

        for child in children:
            child.express_genome()

        return children[0] if len(children) == 1 else children[: n_children]

    def reset_id(self):
        self.id = str(uuid.uuid4())

    def init_genome(self):
        """ Initialises the sequence of the chromosomes randomly. The genes and phenotypes will automatically get
         set/updated from the sequence. """

        for chrom in self.genome.values():
            chr_seq = np.empty(0)
            for gene in chrom.genes.values():
                if gene.encoding in ['real', 'realInt']:
                    chr_seq = np.hstack((chr_seq, np.random.uniform(gene.minv, gene.maxv, size=1)))
                elif gene.encoding in ['binary', 'gray', 'binaryInt', 'grayInt']:
                    chr_seq = np.hstack((chr_seq, np.random.randint(0, 2, size=gene.length)))
                elif gene.encoding == 'categorical':
                    chr_seq = np.hstack((chr_seq, np.random.choice(list(gene.categorical_dict.keys()))))

            chrom.sequence = copy.deepcopy(chr_seq)

        for gene in self.genes.values():
            self.phenotype.update({gene.name: gene.phenotype})

        self.reset_id()  # assign a new unique to the individual
        self.generation = 0  # reset generation

    def express_genome(self):
        """ Reads the genomic sequence of the individual updating its genes and phenotypes. """

        try:
            for chrom in self.genome.values():
                chrom.update_genes()

            for gene in self.genes.values():
                self.phenotype.update({gene.name: gene.phenotype})

        except Exception as e:
            print(f"The genetic information of the individual {self.id} was defective or incomplete. "
                  f"The individual phenotype couldn't be expressed.")

            print('\n', '*' * 30, f"\nGenetic information of individual {self.id}:")
            print('\n', self.genome, '\n', '*' * 30, '\n')
            print(e)

    @staticmethod
    def recombination(chr_p1, chr_p2):

        chr_c1 = copy.deepcopy(chr_p1)
        chr_c2 = copy.deepcopy(chr_p2)

        seq_1 = copy.deepcopy(chr_c1.sequence)
        seq_2 = copy.deepcopy(chr_c2.sequence)

        if chr_p1.length <= 1:
            points = []
        elif chr_p1.recombination == 'single' and np.random.rand() < chr_p1.r_prob:
            points = random.sample(range(chr_p1.length), k=1)

        elif chr_p1.recombination == 'double' and np.random.rand() < chr_p1.r_prob:
            points = random.sample(range(chr_p1.length), k=2)

        elif chr_p1.recombination == 'poisson':
            n_points = np.random.poisson(lam=chr_p1.r_lambda, size=1)[0]
            if n_points:
                while n_points > chr_p1.length:  # the number of recombination points can't be larger than the sequence
                    n_points -= 1
                points = random.sample(range(chr_p1.length), k=n_points)
            else:
                points = []
        else:
            points = []

        for p in points:
            temp = seq_2[:p].copy()
            seq_2[:p], seq_1[:p] = seq_1[:p], temp

        chr_c1.sequence = seq_1
        chr_c2.sequence = seq_2

        return chr_c1, chr_c2

    def show(self):
        pass


class EnvironmentSKL:
    """
    # all selection methods here
    # different types of environment (EnvironmentScikit, Environment...)

    # Available metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

    # - scores: All the results of the GridSeachCV.cv_results_

    # - fitness_func: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """

    def __init__(self, model, X, y, metrics, fitness_func, name=None, kfold=1, test_split=0.33, verbose=1, n_jobs_cv=1):

        if not isinstance(metrics, (tuple, list)):
            metrics = [metrics]

        self.name = name if name else str(uuid.uuid4())
        self.model = model
        self.X = X
        self.y = y
        self.metrics = metrics
        self.fitness_func = fitness_func
        self.kfold = kfold
        self.test_split = test_split  # size of the test set when kfold is set to 1, ignored otherwise.
        self.n_jobs_cv = n_jobs_cv
        self.verbose = verbose

    def evaluate(self, individual, fitness_func=None, kfold=None, n_jobs_cv=None, reuse_scores=False,
                 update_individual=True, verbose=None):
        if not verbose:
            verbose = self.verbose
        if not update_individual:
            individual = copy.deepcopy(individual)
        if not fitness_func:
            fitness_func = self.fitness_func
        if not kfold:
            kfold = self.kfold
        if kfold == 1:
            # GridSearch doesn't allow cv=1, but it can be done passing an instance of ShuffleSplit as cv parameter.
            kfold = ShuffleSplit(test_size=self.test_split, n_splits=1)
        if not n_jobs_cv:
            n_jobs_cv = self.n_jobs_cv

        if reuse_scores and individual.scores:
            pass

        else:
            param_grid = {gene.name: (gene.phenotype,) for gene in individual.genes.values() if not gene.cooperates}

            # cooperative genes
            coop_genes = {}
            for name, gene in individual.genes.items():
                if gene.cooperates:
                    operon = gene.relationship({gene.name: gene.phenotype,
                                                gene.cooperates: individual.genes[gene.cooperates].phenotype})
                    coop_genes.update(operon)

            for k, v in coop_genes.items():
                coop_genes[k] = (v,)  # convert it to tuple for the GridSearchCV

            param_grid.update(coop_genes)

            GCV = GridSearchCV(estimator=self.model, scoring=self.metrics, param_grid=param_grid, cv=kfold,
                               n_jobs=n_jobs_cv, refit=False, verbose=0)
            GCV.fit(self.X, self.y)

            individual.scores = GCV.cv_results_
            individual.fitness = fitness_func(individual)

        return individual


class Environment:
    """
    # all selection methods here
    # different types of environment (EnvironmentScikit, Environment...)

    # Available metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

    # - scores: All the results of the GridSeachCV.cv_results_

    # - fitness_func: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """

    def __init__(self, X, y, fitness_func, X_test=None, y_test=None, name=None, verbose=1, n_jobs_cv=1):

        self.name = name if name else str(uuid.uuid4())
        self.X = X
        self.y = y
        self.fitness_func = fitness_func
        self.n_jobs_cv = n_jobs_cv
        self.verbose = verbose

    def evaluate(self, individual, fitness_func=None, reuse_scores=None, n_jobs_cv=None, update_individual=True,
                 verbose=None):
        if not verbose:
            verbose = self.verbose
        if not update_individual:
            individual = copy.deepcopy(individual)
        if not fitness_func:
            fitness_func = self.fitness_func

        if not n_jobs_cv:
            n_jobs_cv = self.n_jobs_cv

        param_grid = {gene.name: (gene.phenotype,) for gene in individual.genes.values()}

        individual.fitness, individual.scores = fitness_func(self.X, self.y, param_grid)

        return individual


class EnvironmentSKL:
    """
    # all selection methods here
    # different types of environment (EnvironmentScikit, Environment...)

    # Available metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

    # - scores: All the results of the GridSeachCV.cv_results_

    # - fitness_func: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """

    def __init__(self, model, X, y, metrics, fitness_func, name=None, kfold=1, test_split=0.33, verbose=1, n_jobs_cv=1):

        if not isinstance(metrics, (tuple, list)):
            metrics = [metrics]

        self.name = name if name else str(uuid.uuid4())
        self.model = model
        self.X = X
        self.y = y
        self.metrics = metrics
        self.fitness_func = fitness_func
        self.kfold = kfold
        self.test_split = test_split  # size of the test set when kfold is set to 1, ignored otherwise.
        self.n_jobs_cv = n_jobs_cv
        self.verbose = verbose

    def evaluate(self, individual, fitness_func=None, kfold=None, n_jobs_cv=None, reuse_scores=False,
                 update_individual=True, verbose=None):
        if not verbose:
            verbose = self.verbose
        if not update_individual:
            individual = copy.deepcopy(individual)
        if not fitness_func:
            fitness_func = self.fitness_func
        if not kfold:
            kfold = self.kfold
        if kfold == 1:
            # GridSearch doesn't allow cv=1, but it can be done passing an instance of ShuffleSplit as cv parameter.
            kfold = ShuffleSplit(test_size=self.test_split, n_splits=1)
        if not n_jobs_cv:
            n_jobs_cv = self.n_jobs_cv

        if reuse_scores and individual.scores:
            pass

        else:
            param_grid = {gene.name: (gene.phenotype,) for gene in individual.genes.values() if not gene.cooperates}

            # cooperative genes
            coop_genes = {}
            for name, gene in individual.genes.items():
                if gene.cooperates:
                    operon = gene.relationship({gene.name: gene.phenotype,
                                                gene.cooperates: individual.genes[gene.cooperates].phenotype})
                    coop_genes.update(operon)

            for k, v in coop_genes.items():
                coop_genes[k] = (v,)  # convert it to tuple for the GridSearchCV

            param_grid.update(coop_genes)

            GCV = GridSearchCV(estimator=self.model, scoring=self.metrics, param_grid=param_grid, cv=kfold,
                               n_jobs=n_jobs_cv, refit=False, verbose=0)
            GCV.fit(self.X, self.y)

            individual.scores = GCV.cv_results_
            individual.fitness = fitness_func(individual)

        return individual


class Population:
    """
    Make population standard and only Evironment needs to be adapted to the task
    TODO try to make the population as independent from environment as possible so the same populacion
     can be used for eny type of environment (e.g. skleran, custom...)

    - add back-reference on every new created individual?? (individual.population = self)
    """

    # OJO !!! make sure the environment uses gridsearch before hand
    #       if hasattr(obj, 'attr_name'):   <--- This checks if the class as an attribute with that name

    allowed_parallel_modes = ['threading', 'multiprocessing', 'gridsearchCV']
    allowed_selection_methods = ['tournament', 'elitism', 'roulette', 'rank', 'dinamic']  # TODO: Dinamic

    def __init__(self, individual, environment=None, parallelize=False, parallel_mode='threading', n_jobs=-1,
                 reuse_scores=False, name=None, verbose=1):

        if parallel_mode not in parallel_mode:
            raise ValueError(f"parallel_mode '{parallel_mode}' is not allowed (allowed parallel_modes: "
                             f"{self.allowed_parallel_modes}.")

        # TODO: remove battle_size, elitism_size... from here

        self.name = name if name else str(uuid.uuid4())
        self.individual_template = copy.deepcopy(individual)
        self.environment = copy.deepcopy(environment)
        self.verbose = verbose
        self.population = IndexedOrderedDict()
        self.generation = 0
        self.new_generation = IndexedOrderedDict()
        self.temp_population = None
        # self.isPopEval = False  # If the population has already been evaluated

        self.history = pd.DataFrame()
        self.top_individuals = {}

        # reuse_scores: Only relevant when the selection method is 'tournament'.
        # If it is True and the selection method is 'tournament' each warrior will be evaluated only once instead of
        # every time they compete in a tournament. Increases speed significantly but with the cost of a reduction in
        # the entropy of the selection. If eval_once==False due to the randomness of the train/test splits the
        # performance of a warrior can be different of each battle.
        self.reuse_scores = reuse_scores

        # Parallel computing
        self.parallelize = parallelize
        # parallel_mode (options): threading, multiprocessing, gridsearchCV (built-in GridSearchCV multiprocessing)
        self.parallel_mode = parallel_mode
        self.n_CPUs = cpu_count()
        self.n_jobs = self.n_CPUs if (n_jobs == -1 or n_jobs > self.n_CPUs) else n_jobs

        # TODO set n_jobs_cv to none if environment doesn't have a gridsearchCV: hasattr(obj, 'attr_name')
        if self.parallelize and self.parallel_mode == 'gridsearchCV':
            self.n_jobs_cv = self.n_CPUs if (n_jobs == -1 or n_jobs > self.n_CPUs) else n_jobs
        else:
            self.n_jobs_cv = 1

        print("\n", f"Using {self.n_jobs if self.parallelize else 1} / {self.n_CPUs} CPUs", "\n")

    @property
    def pop_size(self):
        return len(self.population)

    def create_individuals(self, n=None):
        if not n:
            n = self.pop_size

        for i in range(n):
            individual = copy.deepcopy(self.individual_template)
            individual.init_genome()
            individual.generation = self.generation

            self.population.update({individual.id: copy.deepcopy(individual)})

    def add(self, individuals):
        # TODO
        # do several check the be sure that the new individuals are compatible (e.g. is instance, length, gen.names...
        pass

    def evaluate(self, individual, fitness_func=None, reuse_scores=None, update_individual=True, verbose=None):
        if not self.environment:
            raise Exception("The population doesn't have any environment yet.")
        if not reuse_scores:
            reuse_scores = self.reuse_scores

        # the ID of the individual can also be passed instead of the individual
        if isinstance(individual, str) and self.population:
            if self.pop_size > 0 and (individual in self.population):
                individual = self.population[individual]
            else:
                raise ValueError(f"The provided ID ({individual}) is not in the population.")

        individual = self.environment.evaluate(individual, fitness_func=fitness_func, reuse_scores=reuse_scores,
                                               update_individual=update_individual, verbose=verbose)
        return individual

    def evaluate_population(self, fitness_func=None, verbose=None, reuse_scores=None):
        if not self.environment:
            raise Exception("The population doesn't have any environment yet.")
        if self.pop_size < 1:
            raise Exception("There is no population to evaluate.")
        if not verbose:
            verbose = self.verbose
        if not reuse_scores:
            reuse_scores = self.reuse_scores

        # TODO: add here before the valuation of the generation 0

        # evaluate all the warriors in the population
        if self.parallelize and self.parallel_mode == 'threading' and self.n_jobs > 1:

            kwd_args = dict(fitness_func=fitness_func, reuse_scores=reuse_scores, verbose=verbose)

            threadPool = ThreadPool(self.n_jobs)
            for individual in tqdm(self.population.values(), disable=1-min(1, verbose)):
                threadPool.apply_async(self.evaluate, args=(individual,), kwds=kwd_args)

            threadPool.close()
            threadPool.join()

        elif self.parallelize and self.parallel_mode == 'multiprocessing' and self.n_jobs > 1:

            pool = Pool(self.n_jobs)
            self.population = pool.map(self.evaluate, self.population.values())  # Todo do something to be able to pass the reuse scores argument for when just eveluating the population for the log_history
            pool.close()
            pool.join()

        else:
            for individual in tqdm(self.population, disable=1-min(1, verbose)):
                self.evaluate(individual, fitness_func=fitness_func, reuse_scores=reuse_scores, verbose=verbose)

        # self.isPopEval = True  # remember to reset to false after evolve

        # gather the evaluation results
        results_eval = []
        for index, individual in enumerate(self.population.values()):
            results_individual = {
                         'generation': individual.generation,
                         'pop_index': index,  # TODO remove
                         'id': individual.id,
                         'fitness': individual.fitness
                         }
            phenotype = {gene.name: gene.phenotype for gene in individual.genes.values()}
            results_individual.update(phenotype)

            results_eval.append(results_individual)

        return pd.DataFrame(results_eval).sort_values(by='fitness', axis=0, ascending=False)

    def battle(self, size=3, groups=2, fitness_func=None, update_individual=True, reuse_scores=None, verbose=None):
        if not self.environment:
            raise Exception("The population doesn't have any environment yet.")
        if not fitness_func:
            fitness_func = self.environment.fitness_func
        if not reuse_scores:
            reuse_scores = self.reuse_scores
        if not verbose:
            verbose = self.verbose

        rankings = []
        groups = np.array_split((random.sample(range(self.pop_size), size*groups)), groups)

        for group in groups:
            warriors = [self.population.values()[w_index] for w_index in group]
            if not update_individual:
                warriors = copy.deepcopy(warriors)

            for w in warriors:

                if reuse_scores and w.fitness:
                    pass

                else:
                    self.evaluate(w, fitness_func=fitness_func, verbose=verbose, reuse_scores=reuse_scores)  # it doesn't matter if update individual it as already mafe a copy (look up)

            # warrior fight classification [1st, 2nd, 3rd, ..., Nth]
            g_ranking = sorted(warriors, key=lambda x: x.fitness, reverse=True)
            rankings.append(g_ranking)

        return rankings[0] if len(rankings) == 1 else rankings

    def breed_warriors(self, *args, fitness_func=None, battle_size=3):
        if not self.environment:
            raise Exception("The population doesn't have any environment yet.")

        if not fitness_func:
            fitness_func = self.environment.fitness_func

        # organize tournament
        ranking_g1, ranking_g2 = self.battle(groups=2, fitness_func=fitness_func, size=battle_size)

        # select winners and breed them
        parent_1, parent_2 = ranking_g1[0], ranking_g2[0]
        children = Individual.from_parents(parent_1, parent_2, n_children=2)

        if self.parallelize and self.parallel_mode == 'multiprocessing':
            return children
        else:
            self.new_generation.update({child.id: child for child in children})

    def tournament_selection(self, save_top_gen=False, battle_size=3, verbose=1, fitness_func=None, log_evo=True):
        if not self.environment:
            raise Exception("The population doesn't have any environment yet.")

        if verbose > 0:
            print(f"\n# Generation {self.generation + 1}")

        save_top_gen = int(save_top_gen)

        if self.parallelize and self.parallel_mode == 'threading' and self.n_jobs > 1:

            kwd_args = dict(fitness_func=fitness_func, battle_size=battle_size)

            threadPool = ThreadPool(self.n_jobs)
            for j in range(self.pop_size // 2):
                threadPool.apply_async(self.breed_warriors, kwds=kwd_args)

            threadPool.close()
            threadPool.join()

        elif self.parallelize and self.parallel_mode == 'multiprocessing' and self.n_jobs > 1:

            pool = Pool(self.n_jobs)
            # Todo do something to be able to pass the "BATTLE_SIZE"
            #  & reuse scores argument for when just eveluating the population for the log_history
            children = pool.map(self.breed_warriors, range(self.pop_size // 2))

            pool.close()
            pool.join()

            children = list(chain(*children))
            self.new_generation.update({child.id: child for child in children})

        else:
            for j in range(self.pop_size // 2):
                self.breed_warriors(fitness_func=fitness_func, battle_size=battle_size)

        # replace the old population with the new generation
        self.population = copy.deepcopy(self.new_generation)
        self.new_generation = IndexedOrderedDict()
        self.generation += 1
        # self.isPopEval = False

        if log_evo or save_top_gen:
            # evaluate the new generation
            gen_eval_df = self.evaluate_population(fitness_func=fitness_func, reuse_scores=True)  # Todo: no hace falte reuse=True, lo que estamos evaluando es la nueva generacion, pero lo dejare por si acaso
            # self.isPopEval = True

            if log_evo:
                self.history = pd.concat([self.history, gen_eval_df])

            if save_top_gen:
                # select the top warriors of the current generation
                gen_top_w_idx = gen_eval_df['pop_index'].iloc[:save_top_gen].values
                gen_top_w = [self.population.items()[i] for i in gen_top_w_idx]
                self.top_individuals[self.generation] = gen_top_w

            if verbose > 1:
                if verbose > 5:
                    print(gen_eval_df.drop(['pop_index', 'generation'], axis=1).head(verbose))
                else:
                    print(gen_eval_df[['generation', 'fitness']].head(verbose))

    def evolve(self, selection_method='elitism', n_gen=1, save_top_gen=False, battle_size=3, elite_size=0.5, verbose=1,
               fitness_func=None, log_evo=True, reuse_scores=None):
        if not self.environment:
            raise Exception("The population doesn't have any environment yet.")

        if not fitness_func:
            fitness_func = self.environment.fitness_func
        if selection_method not in self.allowed_selection_methods:
            raise ValueError(f"parallel_mode '{selection_method}' is not allowed (allowed parallel_modes: "
                             f"{self.allowed_selection_methods}.")
        if not reuse_scores:
            reuse_scores = self.reuse_scores

        save_top_gen = int(save_top_gen)

        for i in range(n_gen):
            # ----------------------------------------------------------------------------------------------------------
            # tournament

            if selection_method == 'tournament':
                self.tournament_selection(save_top_gen=save_top_gen, battle_size=battle_size, verbose=verbose,
                                          fitness_func=fitness_func, log_evo=log_evo)

            # ----------------------------------------------------------------------------------------------------------
            # Others
            else:
                # common (population evaluation) ---------------------------------------------------
                gen_eval_df = self.evaluate_population(verbose=verbose, fitness_func=fitness_func,
                                                       reuse_scores=reuse_scores)
                if log_evo:
                    self.history = pd.concat([self.history, gen_eval_df])

                if save_top_gen:
                    # select the top individuals of the current generation
                    gen_top_idx = gen_eval_df['pop_index'].iloc[:save_top_gen].values
                    gen_top = [self.population.items()[i] for i in gen_top_idx]
                    self.top_individuals[self.generation] = gen_top

                if verbose > 1:
                    if verbose > 5:
                        print(gen_eval_df.drop(['pop_index', 'generation'], axis=1).head(verbose))
                    else:
                        print(gen_eval_df[['generation', 'fitness']].head(verbose))

                if selection_method == 'elitism':
                    # elite breeding ----------------------------------------------------------
                    if elite_size <= 1:  # if elite is between 0-1 it will be interpreted as percentage
                        elite_size = int(self.pop_size * elite_size)

                    elite_idx = gen_eval_df.iloc[:elite_size]['pop_index'].values

                    while len(self.new_generation) < self.pop_size:
                        parents_idx = random.sample(list(elite_idx), k=2)
                        parent_1 = self.population.items()[parents_idx[0]][1]
                        parent_2 = self.population.items()[parents_idx[1]][1]
                        children = Individual.from_parents(p1=parent_1, p2=parent_2, n_children=2)
                        self.new_generation.update({child.id: child for child in children})

                    self.population = copy.deepcopy(self.new_generation)
                    self.new_generation = IndexedOrderedDict()
                    self.generation += 1

                elif selection_method == 'roulette':
                    # roulette wheel breeding ---------------------------------------------------
                    while len(self.new_generation) < self.pop_size:
                        parents_idx = self.roulette_wheel(eval_df=gen_eval_df, n=2)
                        parent_1 = self.population.items()[parents_idx[0]][1]
                        parent_2 = self.population.items()[parents_idx[1]][1]

                        children = Individual.from_parents(p1=parent_1, p2=parent_2, n_children=2)
                        self.new_generation.update({child.id: child for child in children})

                    self.population = copy.deepcopy(self.new_generation)
                    self.new_generation = IndexedOrderedDict()
                    self.generation += 1

                elif selection_method == 'rank':
                    # rank selection breeding -----------------------------------------------------
                    while len(self.new_generation) < self.pop_size:
                        parents_idx = self.rank_selection(eval_df=gen_eval_df, n=2)
                        parent_1 = self.population.items()[parents_idx[0]][1]
                        parent_2 = self.population.items()[parents_idx[1]][1]

                        children = Individual.from_parents(p1=parent_1, p2=parent_2, n_children=2)
                        self.new_generation.update({child.id: child for child in children})

                    self.population = copy.deepcopy(self.new_generation)
                    self.new_generation = IndexedOrderedDict()
                    self.generation += 1

                # common (last generation) -------------------------------------------------------------
                if i == n_gen - 1:
                    if save_top_gen or log_evo:
                        gen_eval_df = self.evaluate_population(verbose=verbose, fitness_func=fitness_func)

                        if log_evo:
                            self.history = pd.concat([self.history, gen_eval_df])

                        if save_top_gen:
                            # select the top individuals of the current generation
                            gen_top_idx = gen_eval_df['pop_index'].iloc[:save_top_gen].values
                            gen_top = [self.population.items()[i] for i in gen_top_idx]
                            self.top_individuals[self.generation] = gen_top

                        if verbose > 1:
                            if verbose > 5:
                                print(gen_eval_df.drop(['pop_index', 'generation'], axis=1).head(verbose))
                            else:
                                print(gen_eval_df[['generation', 'fitness']].head(verbose))

        if save_top_gen or log_evo:
            history = copy.deepcopy(self.history)

            return (history, self.top_individuals) if save_top_gen else history

    @staticmethod
    def roulette_wheel(eval_df, n=1):
        n = int(n)
        eval_df = eval_df.copy()
        min_f = eval_df['fitness'].min()

        if min_f < 0:
            eval_df['fitness'] = eval_df['fitness'] + abs(min_f)  # move up fitness to positive values

        sum_f = eval_df['fitness'].sum()
        points = sorted([np.random.uniform(0, sum_f) for i in range(n)])

        selected = []
        for P in points:
            current_fitness = 0
            for i in range(eval_df.shape[0]):
                current_fitness += eval_df.loc[i, 'fitness']
                if current_fitness >= P:
                    selected.append(eval_df.loc[i, 'pop_index'])
                    break
        return selected

    @staticmethod
    def rank_selection(eval_df, n=1):
        n = int(n)
        eval_df = eval_df.copy()
        eval_df.sort_values(by='fitness', axis=0, inplace=True)
        eval_df.reset_index()

        pop_length = eval_df.shape[0]
        probabilities = []

        for i in range(pop_length + 1):
            if i == 0:
                probabilities.append(0)
            else:
                next_proba = (i + 1) / pop_length
                probabilities.append(probabilities[i - 1] + (next_proba / pop_length))

        selected = []
        while len(selected) < n:
            random_number = np.random.rand()
            for j in range(pop_length):
                if random_number < probabilities[j + 1]:
                    selected.append(eval_df.iloc[j]['pop_index'])
                    break
        return selected[0] if len(selected) == 1 else selected


class PopulationFS:
    #TODO
    pass


# Old

# class Population:
#     """ """
#
#     def __init__(self, size, genes, environment, kfold=1, battle_size=5, prob_c=0.75, prob_c_s=0.33, prob_m=0.1,
#                  eval_once=False, parallelize=True, parallel_mode='threading', test_split=0.33,
#                  n_jobs=-1):
#         self.pop_size = int(size)
#         self.start = 0  # iterator
#         self.generation = 0
#         self.genes = IndexedOrderedDict({gene['name']: gene for gene in genes})  # empty template of the warrior's genome
#         self.environment = environment
#         self.kfold = kfold
#         self.battle_size = battle_size
#         self.prob_c = prob_c  # probability of chromosomal crossover (either single or double point)
#         self.prob_c_s = prob_c_s  # probability of single point crossover
#         self.prob_m = prob_m  # probability of mutation
#         self.test_split = test_split  # size of the test set when kfold is set to 1, ignored otherwise.
#
#         # Parallel computing
#         self.parallelize = parallelize
#         # parallel_mode (options): threading, multiprocessing, gridsearchCV (built-in GridSearchCV multiprocessing)
#         self.parallel_mode = parallel_mode
#         self.n_CPUs = cpu_count()
#         self.n_jobs = self.n_CPUs if (n_jobs == -1 or n_jobs > self.n_CPUs) else n_jobs
#         if self.parallelize and self.parallel_mode == 'gridsearchCV':
#             self.n_jobs_CV = self.n_CPUs if (n_jobs == -1 or n_jobs > self.n_CPUs) else n_jobs
#         else:
#             self.n_jobs_CV = 1
#
#         print("\n", f"Using {self.n_jobs if self.parallelize else 1} / {self.n_CPUs} CPUs", "\n")
#
#         # eval_once: If True each warrior will be evaluated only once instead of every time they compete in a
#         # tournament. Increases speed significantly but with the cost of a reduction in the entropy of the
#         # selection. If eval_once==False due to the randomness of the train/test splits the performance of a
#         # warrior can be different of each battle.
#         self.eval_once = eval_once
#
#         index = 0
#         for gene in self.genes.keys():
#
#             if self.genes[gene].get('len'):
#                 # calculate the precision
#                 self.genes[gene]['precision'] = self.calc_precision(self.genes[gene])
#             elif self.genes[gene].get('precision'):
#                 # calculate the length of the encoded binary
#                 self.genes[gene]['len'] = self.calc_length_binary(self.genes[gene])
#             else:
#                 raise TypeError("Neither gene['len'] or gene['precision'] was provided. At least one of them"
#                                 "must be specified.")
#
#             # map the location of the genes in the chromosome
#             self.genes[gene]['location'] = (index, (index + self.genes[gene]['len']))
#             index += self.genes[gene]['len']
#
#         # create first generation
#         self.population = self.create_warriors(n=self.pop_size)
#         self.temp_population = None
#         self.new_generation = []
#         self.isEvaluated = False
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.start >= len(self.population):
#             raise StopIteration
#         else:
#             self.start += 1
#             return self.population[self.start - 1]
#
#     @staticmethod
#     def calc_precision(gene):
#         return (gene['max'] - gene['min']) / ((2 ** gene['len']) - 1)
#
#     @staticmethod
#     def calc_length_binary(gene):
#         return int(np.ceil(np.log2(((gene['max'] - gene['min']) / gene['precision']) - 1)))
#
#     @staticmethod
#     def encode_gen(gene):
#         pass  # TODO
#
#     @staticmethod
#     def decode_gen(gene):
#         z = 0
#         val = 0
#         for i in range(gene['len'] - 1, -1, -1):
#             vi = gene['genotype'][i] * 2 ** z
#             val += vi
#             z = z + 1
#         phenotype = (gene['precision'] * val) + gene['min']
#         return phenotype
#
#     def create_warriors(self, n):
#         population = []
#         for i in range(n):
#             warrior = {
#                 'id': str(uuid.uuid4()),
#                 'generation': self.generation,
#                 'genes': copy.deepcopy(self.genes),
#                 'chromosome': np.empty(0),
#                 'scores': None,
#                 'fitness': None
#             }
#             # initialize genome randomly
#             for gene in self.genes.keys():
#                 genotype = np.random.randint(0, 2, size=(self.genes[gene]['len']))
#                 warrior['genes'][gene]['genotype'] = genotype
#                 warrior['genes'][gene]['phenotype'] = self.decode_gen(warrior['genes'][gene])
#                 warrior['chromosome'] = np.hstack((warrior['chromosome'], genotype))
#
#             population.append(warrior)
#
#         return population
#
#     def read_chr(self, chr):
#         genes = copy.deepcopy(self.genes)  # empty template of the genes
#         for gene in genes.keys():
#             loc_ini, loc_end = genes[gene]['location']  # location of the gene in the chromosome
#             genotype = chr[loc_ini: loc_end]  # read the that location from the warrior's chromosome
#             genes[gene]['genotype'] = genotype
#             genes[gene]['phenotype'] = self.decode_gen(genes[gene])  # decode the genetic binary information
#
#         return genes
#
#     def battle_royal(self, size=None, groups=1, fitness_func=None, kfold=None):
#         if not fitness_func:
#             fitness_func = self.environment['fitness_func']
#         if not size:
#             size = self.battle_size
#         if not kfold:
#             kfold = self.kfold
#         if kfold == 1:
#             # GridSearch doesn't allow cv=1, but it can be done passing an instance of ShuffleSplit as cv parameter.
#             kfold = ShuffleSplit(test_size=self.test_split, n_splits=1)
#
#         rankings = []
#         groups = np.array_split((random.sample(range(len(self.population)), size)), groups)
#
#         for group in groups:
#             warriors = [self.population[w_index] for w_index in group]
#             warriors = copy.deepcopy(warriors)
#
#             for w in warriors:
#
#                 if self.eval_once and w['fitness']:
#                     pass
#
#                 else:
#                     param_grid = {w['genes'][gene]['name']: (w['genes'][gene]['phenotype'],) for gene in w['genes'].keys()}
#
#                     GCV = GridSearchCV(estimator=self.environment['model'],
#                                        scoring=self.environment['scores'],
#                                        param_grid=param_grid, cv=kfold, n_jobs=self.n_jobs_CV, refit=False)
#                     GCV.fit(self.environment['X'], self.environment['y'])
#
#                     w['scores'] = GCV.cv_results_
#                     w['fitness'] = fitness_func(w)
#
#             # warrior fight classification [1st, 2nd, 3rd, ..., Nth]
#             g_ranking = sorted(warriors, key=lambda x: x['fitness'], reverse=True)
#             rankings.append(g_ranking)
#
#         return rankings[0] if len(rankings) == 1 else rankings
#
#     def make_children(self, p1, p2, n=2):
#         chr_c1 = copy.deepcopy(p1['chromosome'])
#         chr_c2 = copy.deepcopy(p2['chromosome'])
#
#         children = []
#         for i in range(int(np.ceil(n/2))):
#             crossover = np.random.rand()  # chromosomal crossover chance
#             single_cross = np.random.rand()  # chance on the crossover being "single point"
#
#             if crossover < self.prob_c:
#                 if single_cross < self.prob_c_s:
#                     cut = np.random.randint(0, len(chr_c1))
#                     tmp = chr_c2[:cut]
#                     chr_c2[:cut], chr_c1[:cut] = chr_c1[:cut], tmp
#
#                 else:
#                     cut1, cut2 = sorted(random.sample(range(len(chr_c1)), 2))
#                     tmp = chr_c2[cut1:cut2]
#                     chr_c2[cut1:cut2], chr_c1[cut1:cut2] = chr_c1[cut1:cut2], tmp
#
#             for chr in [chr_c1, chr_c2]:
#
#                 # mutations
#                 for j in range(len(chr)):
#                     mutation = np.random.rand()  # mutation chance
#                     if mutation < self.prob_m:
#                         chr[j] = 1 - chr[j]  # mutate by swapping the allele value
#
#                 child = {
#                     'id': str(uuid.uuid4()),
#                     'generation': self.generation + 1,
#                     'chromosome': chr,
#                     'genes': self.read_chr(chr),
#                     'scores': None,
#                     'fitness': None
#                 }
#                 children.append(child)
#
#         return children[:n]
#
#     def evaluate_warrior(self, warrior, fitness_func=None, kfold=None, reuse_scores=False):
#         if not fitness_func:
#             fitness_func = self.environment['fitness_func']
#         if not kfold:
#             kfold = self.kfold
#         if kfold == 1:
#             # GridSearch doesn't allow cv=1, but it can be done passing an instance of ShuffleSplit as cv parameter.
#             kfold = ShuffleSplit(test_size=self.test_split, n_splits=1)
#
#         if (self.eval_once or reuse_scores) and warrior['scores']:
#             pass
#
#         else:
#             param_grid = {warrior['genes'][gene]['name']: (warrior['genes'][gene]['phenotype'],) for gene in warrior['genes'].keys()}
#
#             GCV = GridSearchCV(estimator=self.environment['model'],
#                                scoring=self.environment['scores'],
#                                param_grid=param_grid, cv=kfold, n_jobs=self.n_jobs_CV, refit=False)
#             GCV.fit(self.environment['X'], self.environment['y'])
#
#             warrior['scores'] = GCV.cv_results_
#             warrior['fitness'] = fitness_func(warrior)
#
#         return warrior
#
#     def evaluate_population(self, fitness_func=None, kfold=None, reuse_scores=False, verbose=0):
#         if not fitness_func:
#             fitness_func = self.environment['fitness_func']
#         if not kfold:
#             kfold = self.kfold
#
#         # evaluate all the warriors in the population
#         if self.parallelize and self.parallel_mode == 'threading' and self.n_jobs > 1:
#
#             kwd_args = dict(fitness_func=fitness_func, kfold=kfold)
#
#             threadPool = ThreadPool(self.n_jobs)
#             for w in tqdm(self.population, disable=1-min(1, verbose)):
#                 threadPool.apply_async(self.evaluate_warrior, args=(w,), kwds=kwd_args)
#
#             threadPool.close()
#             threadPool.join()
#
#         elif self.parallelize and self.parallel_mode == 'multiprocessing' and self.n_jobs > 1:
#
#             pool = Pool(self.n_jobs)
#             self.population = pool.map(self.evaluate_warrior, self.population)
#             pool.close()
#             pool.join()
#
#         else:
#             for w in tqdm(self.population, disable=1-min(1, verbose)):
#                 self.evaluate_warrior(w, fitness_func=fitness_func, kfold=kfold)
#
#         self.isEvaluated = True
#
#         # gather the evaluation results
#         results_eval = []
#         for i, w in enumerate(self.population):
#             w_results = {
#                          'generation': self.generation,
#                          'pop_index': i,
#                          'id': w['id'],
#                          'fitness': w['fitness']
#                          }
#             phenotype = {gene: w['genes'][gene]['phenotype'] for gene in w['genes'].keys()}
#             w_results.update(phenotype)
#
#             results_eval.append(w_results)
#
#         return pd.DataFrame(results_eval).sort_values(by='fitness', axis=0, ascending=False)
#
#     def breed(self, *args, fitness_func=None, kfold=None):
#         if not fitness_func:
#             fitness_func = self.environment['fitness_func']
#         if not kfold:
#             kfold = self.kfold
#
#         # organize tournament
#         ranking_g1, ranking_g2 = self.battle_royal(groups=2, fitness_func=fitness_func, kfold=kfold)
#
#         # select winners and breed them
#         parent_1, parent_2 = ranking_g1[0], ranking_g2[0]
#         children = self.make_children(parent_1, parent_2, n=2)
#
#         if self.parallelize and self.parallel_mode == 'multiprocessing':
#             return children
#         else:
#             self.new_generation += children
#
#     def evolve(self, n_gen=1, save_top_gen=False, kfold=None, verbose=5, fitness_func=None):
#         if not fitness_func:
#             fitness_func = self.environment['fitness_func']
#         if not kfold:
#             kfold = self.kfold
#
#         evo_log_df = pd.DataFrame()
#         top_w = {}  # Only used if save_top_gen True
#
#         for i in range(n_gen):
#             if verbose > 0:
#                 print(f"\n# Generation {self.generation + 1}")
#
#             if self.parallelize and self.parallel_mode == 'threading' and self.n_jobs > 1:
#
#                 threadPool = ThreadPool(self.n_jobs)
#                 for j in range(self.pop_size // 2):
#                     threadPool.apply_async(self.breed, args=(fitness_func, kfold))
#
#                 threadPool.close()
#                 threadPool.join()
#
#             elif self.parallelize and self.parallel_mode == 'multiprocessing' and self.n_jobs > 1:
#
#                 pool = Pool(self.n_jobs)
#                 children = pool.map(self.breed, range(self.pop_size // 2))
#
#                 pool.close()
#                 pool.join()
#
#                 self.new_generation = list(chain(*children))  # flatten
#
#             else:
#                 for j in range(self.pop_size // 2):
#                     self.breed(fitness_func=fitness_func, kfold=kfold)
#
#             # replace the old population with the new generation
#             self.population = copy.deepcopy(self.new_generation)
#             self.new_generation = []
#             self.generation += 1
#             self.isEvaluated = False
#
#             # evaluate the new generation
#             gen_eval_df = self.evaluate_population(kfold=kfold, fitness_func=fitness_func)
#
#             evo_log_df = pd.concat([evo_log_df, gen_eval_df])
#
#             if save_top_gen:
#                 # select the top warriors of the current generation
#                 gen_top_w_idx = gen_eval_df['pop_index'].iloc[:save_top_gen].values
#                 gen_top_w = [self.population[i] for i in gen_top_w_idx]
#                 top_w[self.generation] = gen_top_w
#
#             if verbose > 1:
#                 if verbose > 5:
#                     print(gen_eval_df.drop(['pop_index', 'generation'], axis=1).head(verbose))
#                 else:
#                     print(gen_eval_df[['generation', 'fitness']].head(verbose))
#
#         if save_top_gen:
#             return evo_log_df, top_w
#
#         return evo_log_df
