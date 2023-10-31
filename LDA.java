/*
 * Copyright 2011 Arnim Bleier, Andreas Niekler and Patrick Jaehnichen
 * Licensed under the GNU Lesser General Public License.
 * http://www.gnu.org/licenses/lgpl.html
 */

package de.uni_leipzig.informatik.asv.hdp;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import de.uni_leipzig.informatik.asv.utils.*;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.stream.IntStream;
/**
 * Hierarchical Dirichlet Processes  
 * Chinese Restaurant Franchise Sampler
 * 
 * For more information on the algorithm see:
 * Hierarchical Bayesian Nonparametric Models with Applications. 
 * Y.W. Teh and M.I. Jordan. Bayesian Nonparametrics, 2010. Cambridge University Press.
 * http://www.gatsby.ucl.ac.uk/~ywteh/research/npbayes/TehJor2010a.pdf
 * 
 * For other known implementations see README.txt
 * 
 * @author <a href="mailto:arnim.bleier+hdp@gmail.com">Arnim Bleier</a>
 */
public class LDA { 

	// public double beta  = 0.5; // default only
	// public double alpha = 0.5;

	private Random random = new Random();
	
	protected DOCState[] docStates;
	protected int[] numberOfTablesByTopic;
	protected int[] wordCountByTopic;
	protected int[][] wordCountByTopicAndTerm;
	protected int[][] wordCountByDocAndTopic;
	
	
	protected int sizeOfVocabulary;
	protected int totalNumberOfWords;
	public int numberOfTopics = 10;

	public double beta  = 1/ Double.valueOf(numberOfTopics); // default only
	public double alpha = 1/ Double.valueOf(numberOfTopics);
	/**
	 * Initially assign the words to tables and topics
	 * 
	 * @param corpus {@link CLDACorpus} on which to fit the model
	 */
	public void addInstances(int[][] documentsInput, int V) {

		sizeOfVocabulary = V;
		totalNumberOfWords = 0;
		docStates = new DOCState[documentsInput.length];
		for (int d = 0; d < documentsInput.length; d++) {
			docStates[d] = new DOCState(documentsInput[d], d);
			totalNumberOfWords += documentsInput[d].length;
		}
		int k, i, j, d;
		DOCState docState;
		
		// numberOfTablesByTopic = new int[numberOfTopics];
		wordCountByTopic = new int[numberOfTopics];
		wordCountByTopicAndTerm = new int[numberOfTopics][];
		for (k = 0; k < numberOfTopics; k++) 	// var initialization done
			wordCountByTopicAndTerm[k] = new int[sizeOfVocabulary];
		
		wordCountByDocAndTopic = new int[documentsInput.length][];
		for (d = 0; d < documentsInput.length; d++) 	// var initialization done
			wordCountByDocAndTopic[d] = new int[numberOfTopics];

		// for (k = 0; k < numberOfTopics; k++) { 
		// 	docState = docStates[k];
		// 	for (i = 0; i < docState.documentLength; i++) 
		// 		addWord(docState.docID, i, k);
		// } // all topics have now one document
		for (j = 0; j < docStates.length; j++) {
			docState = docStates[j]; 
			for (i = 0; i < docState.documentLength; i++) {
				k = random.nextInt(numberOfTopics);
				addWord(docState.docID, i, k);
			}
		} // the words in the remaining documents are now assigned too
	}

	
	/**
	 * Step one step ahead
	 * 
	 */
	protected void nextGibbsSweep() {
		int topic;
		for (int d = 0; d < docStates.length; d++) {
			for (int i = 0; i < docStates[d].documentLength; i++) {
				removeWord(d, i); // remove the word i from the state
				topic = sampleTopic(d, i);
				addWord(d, i, topic); // existing Table
			}
		}
		// defragment();
	}
	

	/**	 
	 * Decide at which table the word should be assigned to
	 * 
	 * @param docID the index of the document of the current word
	 * @param i the index of the current word
	 * @return the index of the table
	 */
	int sampleTopic(int docID, int i) {	
		int k, j;
		double pSum = 0.0, vb = sizeOfVocabulary * beta, u;
		DOCState docState = docStates[docID];
		double[] p = new double[numberOfTopics];
		for (k = 0; k < numberOfTopics; k++) {
			pSum += (wordCountByTopicAndTerm[k][docState.words[i].termIndex] + beta) * (wordCountByDocAndTopic[docID][k] + alpha) / 
					(wordCountByTopic[k] + vb);
			p[k] = pSum;
		}

		u = random.nextDouble() * pSum;
		for (j = 0; j < numberOfTopics; j++)
			if (u < p[j]) 
				break;	// decided which table the word i is assigned to
		return j;
	}


	/**
	 * Method to call for fitting the model.
	 * 
	 * @param doShuffle
	 * @param shuffleLag
	 * @param maxIter number of iterations to run
	 * @param saveLag save interval 
	 * @param wordAssignmentsWriter {@link WordAssignmentsWriter}
	 * @param topicsWriter {@link TopicsWriter}
	 * @throws IOException 
	 */
	public void run(int shuffleLag, int maxIter, PrintStream log) 
	throws IOException {
		for (int iter = 0; iter < maxIter; iter++) {
			if ((shuffleLag > 0) && (iter > 0) && (iter % shuffleLag == 0))
				doShuffle();
			nextGibbsSweep();
			log.println("iter = " + iter + " #topics = " + numberOfTopics);
		}
	}
		
	
	/**
	 * Removes a word from the bookkeeping
	 * 
	 * @param docID the id of the document the word belongs to 
	 * @param i the index of the word
	 */
	protected void removeWord(int docID, int i){
		DOCState docState = docStates[docID];
		int k = docState.words[i].topicAssignment;
		wordCountByTopic[k]--; 		
		wordCountByTopicAndTerm[k][docState.words[i].termIndex] --;
		wordCountByDocAndTopic[docID][k] --;
		// if (docState.wordCountByTable[table] == 0) { // table is removed
		// 	totalNumberOfTables--; 
		// 	numberOfTablesByTopic[k]--; 
		// 	docState.tableToTopic[table] --; 
		// }
	}
	
	
	
	/**
	 * Add a word to the bookkeeping
	 * 
	 * @param docID	docID the id of the document the word belongs to 
	 * @param i the index of the word
	 * @param table the table to which the word is assigned to
	 * @param k the topic to which the word is assigned to
	 */
	protected void addWord(int docID, int i, int k) {
		DOCState docState = docStates[docID];
		docState.words[i].topicAssignment = k; 
		wordCountByTopic[k]++; 
		wordCountByTopicAndTerm[k][docState.words[i].termIndex] ++;
		wordCountByDocAndTopic[docID][k] ++;
		// if (docState.wordCountByTable[table] == 1) { // a new table is created
		// 	docState.numberOfTables++;
		// 	docState.tableToTopic[table] = k;
		// 	totalNumberOfTables++;
		// 	numberOfTablesByTopic[k]++; 
		// 	docState.tableToTopic = ensureCapacity(docState.tableToTopic, docState.numberOfTables);
		// 	docState.wordCountByTable = ensureCapacity(docState.wordCountByTable, docState.numberOfTables);
		// 	if (k == numberOfTopics) { // a new topic is created
		// 		numberOfTopics++; 
		// 		numberOfTablesByTopic = ensureCapacity(numberOfTablesByTopic, numberOfTopics); 
		// 		wordCountByTopic = ensureCapacity(wordCountByTopic, numberOfTopics);
		// 		wordCountByTopicAndTerm = add(wordCountByTopicAndTerm, new int[sizeOfVocabulary], numberOfTopics);
		// 	}
		// }
	}

	
	/**
	 * Removes topics from the bookkeeping that have no words assigned to
	 */
	protected void defragment() {
		int[] kOldToKNew = new int[numberOfTopics];
		int k, newNumberOfTopics = 0;
		for (k = 0; k < numberOfTopics; k++) {
			if (wordCountByTopic[k] > 0) {
				kOldToKNew[k] = newNumberOfTopics;
				swap(wordCountByTopic, newNumberOfTopics, k);
				swap(numberOfTablesByTopic, newNumberOfTopics, k);
				swap(wordCountByTopicAndTerm, newNumberOfTopics, k);
				newNumberOfTopics++;
			} 
		}
		numberOfTopics = newNumberOfTopics;
		for (int j = 0; j < docStates.length; j++) 
			docStates[j].defragment(kOldToKNew);
	}
	
	
	/**
	 * Permute the ordering of documents and words in the bookkeeping
	 */
	protected void doShuffle(){
		List<DOCState> h = Arrays.asList(docStates);
		Collections.shuffle(h);
		docStates = h.toArray(new DOCState[h.size()]);
		for (int j = 0; j < docStates.length; j ++){
			List<WordState> h2 = Arrays.asList(docStates[j].words);
			Collections.shuffle(h2);
			docStates[j].words = h2.toArray(new WordState[h2.size()]);
		}
	}
	
	
	
	public static void swap(int[] arr, int arg1, int arg2){
		   int t = arr[arg1]; 
		   arr[arg1] = arr[arg2]; 
		   arr[arg2] = t; 
	}
	
	public static void swap(int[][] arr, int arg1, int arg2) {
		   int[] t = arr[arg1]; 
		   arr[arg1] = arr[arg2]; 
		   arr[arg2] = t; 
	}
	
	public static double[] ensureCapacity(double[] arr, int min){
		int length = arr.length;
		if (min < length)
			return arr;
		double[] arr2 = new double[min*2];
		for (int i = 0; i < length; i++) 
			arr2[i] = arr[i];
		return arr2;
	}

	public static int[] ensureCapacity(int[] arr, int min) {
		int length = arr.length;
		if (min < length)
			return arr;
		int[] arr2 = new int[min*2];
		for (int i = 0; i < length; i++) 
			arr2[i] = arr[i];
		return arr2;
	}

	public static int[][] add(int[][] arr, int[] newElement, int index) {
		int length = arr.length;
		if (length <= index){
			int[][] arr2 = new int[index*2][];
			for (int i = 0; i < length; i++) 
				arr2[i] = arr[i];
			arr = arr2;
		}
		arr[index] = newElement;
		return arr;
	}
	
	

	
	class DOCState {
		
		int docID, documentLength, numberOfTables;
		int[] tableToTopic; 
	    int[] wordCountByTable;
		WordState[] words;

		public DOCState(int[] instance, int docID) {
			this.docID = docID;
		    numberOfTables = 0;  
		    documentLength = instance.length;
		    words = new WordState[documentLength];	
		    wordCountByTable = new int[2];
		    tableToTopic = new int[2];
			for (int position = 0; position < documentLength; position++) 
				words[position] = new WordState(instance[position], -1);
		}


		public void defragment(int[] kOldToKNew) {
		    int[] tOldToTNew = new int[numberOfTables];
		    int t, newNumberOfTables = 0;
		    for (t = 0; t < numberOfTables; t++){
		        if (wordCountByTable[t] > 0){
		            tOldToTNew[t] = newNumberOfTables;
		            tableToTopic[newNumberOfTables] = kOldToKNew[tableToTopic[t]];
		            swap(wordCountByTable, newNumberOfTables, t);
		            newNumberOfTables ++;
		        } else 
		        	tableToTopic[t] = -1;
		    }
		    numberOfTables = newNumberOfTables;
		    for (int i = 0; i < documentLength; i++)
		        words[i].topicAssignment = tOldToTNew[words[i].topicAssignment];
		}

	}
	
	
	class WordState {   
	
		int termIndex;
		int topicAssignment;
		
		public WordState(int wordIndex, int topicAssignment){
			this.termIndex = wordIndex;
			this.topicAssignment = topicAssignment;
		}

	}
	
	
	public static void main(String[] args) throws IOException {

		ArrayList<String> datasets = new ArrayList<String>();
		datasets.add("20news");
		datasets.add("TagMyNews");
		datasets.add("Dbpedia14");
		datasets.add("TwitterEmotion");
		// datasets.add("Yelp");
		datasets.add("AGNews");

		ArrayList<Integer> topic_nums = new ArrayList<Integer>();
		// topic_nums.add(10);
		topic_nums.add(20);
		// topic_nums.add(50);
		// topic_nums.add(100);
		// topic_nums.add(200);


		for (String dataset : datasets){
			InputStream is = new FileInputStream("HDP/src/main/java/de/uni_leipzig/informatik/asv/hdp/01062023/vobs_" + dataset + "_oov.txt");
			BufferedReader br = new BufferedReader(new InputStreamReader(is,"UTF-8"));
			String line = null;
			List<String> vobs = new ArrayList<String>();
			while ((line = br.readLine()) != null) {
				vobs.add(line);
			}

			CLDACorpus corpus = new CLDACorpus(new FileInputStream("HDP/src/main/java/de/uni_leipzig/informatik/asv/hdp/01062023/data_" + dataset + "_train_oov.txt"));
			
			for (Integer topic_num : topic_nums){
				LDA hdp = new LDA();
				hdp.numberOfTopics = topic_num;
				hdp.addInstances(corpus.getDocuments(), corpus.getVocabularySize());

				System.out.println("sizeOfVocabulary = "+hdp.sizeOfVocabulary);
				System.out.println("totalNumberOfWords = "+hdp.totalNumberOfWords);
				System.out.println("NumberOfDocs = "+hdp.docStates.length);

				hdp.run(0, 2000, System.out);
				

				// PrintStream file = new PrintStream(args[1]);
				PrintStream file = new PrintStream("/Users/zhengfang/Library/CloudStorage/OneDrive-Personal/Projects/Topic-SUM/results/LDA/" + dataset + "/oov_LDA5_" + hdp.numberOfTopics + "topics_nzw.txt");
				for (int k = 0; k < hdp.numberOfTopics; k++) {
					for (int w = 0; w < hdp.sizeOfVocabulary; w++)
						file.format("%05d ",hdp.wordCountByTopicAndTerm[k][w]);
					file.println();
				}
				file.close();

				for (int k = 0; k < hdp.numberOfTopics; k++) {
					List<String> word = new ArrayList<String>();
					List<Integer> topic_k = new ArrayList<Integer>();
					for (int i : hdp.wordCountByTopicAndTerm[k])
					{
						topic_k.add(i);
					}
					for (int i = 0; i<20; i++){
						int max_index = IntStream.range(0, hdp.sizeOfVocabulary).boxed().max(Comparator.comparingInt(ix -> topic_k.get(ix))).get();
						word.add(vobs.get(max_index));
						topic_k.set(max_index, 0);
					}
					System.out.println(word);
				}

				// file = new PrintStream(args[2]); /
				// file = new PrintStream("HDP/src/main/java/de/uni_leipzig/informatik/asv/hdp/LDA_20topics.txt");
				file = new PrintStream("/Users/zhengfang/Library/CloudStorage/OneDrive-Personal/Projects/Topic-SUM/results/LDA/" + dataset + "/oov_LDA5_" + hdp.numberOfTopics + "topics.txt");
				file.println("d w z");
				int z, docID;
				for (int d = 0; d < hdp.docStates.length; d++) {
					DOCState docState = hdp.docStates[d];
					docID = docState.docID;
					for (int i = 0; i < docState.documentLength; i++) {
						z = docState.words[i].topicAssignment;
						file.println(docID + " " + docState.words[i].termIndex + " " + z); 
					}
				}
				file.close();
			}
		}
	}
}