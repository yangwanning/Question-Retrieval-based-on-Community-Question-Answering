#include "indri/ScoredExtentResult.hpp"
#include "indri/QueryEnvironment.hpp"
#include "indri/QueryParserFactory.hpp"
#include "indri/QueryAnnotation.hpp"
#include "indri/Parameters.hpp"
#include <math.h>
#include <string>
#include <ctype.h>
#include <ctime>
#include <sstream>
#include <queue>
#include <time.h>
#include "indri/LocalQueryServer.hpp"
#include "indri/delete_range.hpp"
#include "indri/NetworkStream.hpp"
#include "indri/NetworkMessageStream.hpp"
#include "indri/NetworkServerProxy.hpp"
#include "indri/ListIteratorNode.hpp"
#include "indri/ExtentInsideNode.hpp"
#include "indri/DocListIteratorNode.hpp"
#include "indri/FieldIteratorNode.hpp"
#include "indri/ParsedDocument.hpp"
#include "indri/Collection.hpp"
#include "indri/CompressedCollection.hpp"
#include "indri/TaggedDocumentIterator.hpp"
#include "indri/XMLNode.hpp"
#include "indri/QueryExpander.hpp"
#include "indri/RMExpander.hpp"
#include "indri/PonteExpander.hpp"
#include "indri/TFIDFExpander.hpp"
#include "indri/IndriTimer.hpp"
#include "indri/UtilityThread.hpp"
#include "indri/ScopedLock.hpp"
#include "indri/delete_range.hpp"
#include "indri/SnippetBuilder.hpp"
#include "indri/Porter_Stemmer.hpp"

using namespace lemur::api;
using namespace indri::api;

bool isNotAlpha(char c)
{
	return !isalpha(c);
}

/*
 * Code snippet copied from IndriRunQuery.cpp
 */

struct query_t {
	struct greater {
		bool operator() ( query_t* one, query_t* two ) {
			return one->index > two->index;
		}
	};

	query_t( int _index, std::string _number, const std::string& _text, const std::string &queryType,  std::vector<std::string> workSet,   std::vector<std::string> FBDocs) :
		index( _index ),
		number( _number ),
		text( _text ), qType(queryType), workingSet(workSet), relFBDocs(FBDocs)
	{
	}

	query_t( int _index, std::string _number, const std::string& _text ) :
		index( _index ),
		number( _number ),
		text( _text )
	{
	}

	std::string number;
	int index;
	std::string text;
	std::string qType;
	// working set to restrict retrieval
	std::vector<std::string> workingSet;
	// Rel fb docs
	std::vector<std::string> relFBDocs;
};


/*
 * code snippet copied as-is from IndriRunQuery.cpp
 */
void push_queue( std::queue< query_t* >& q, indri::api::Parameters& queries,int queryOffset ) {

	for( size_t i=0; i<queries.size(); i++ ) {
		std::string queryNumber;
		std::string queryText;
		std::string queryType = "indri";
		if( queries[i].exists( "type" ) )
			queryType = (std::string) queries[i]["type"];
		if (queries[i].exists("text"))
			queryText = (std::string) queries[i]["text"];
		if( queries[i].exists( "number" ) ) {
			queryNumber = (std::string) queries[i]["number"];
		} else {
			int thisQuery=queryOffset + int(i);
			std::stringstream s;
			s << thisQuery;
			queryNumber = s.str();
		}
		if (queryText.size() == 0)
			queryText = (std::string) queries[i];

		// working set and RELFB docs go here.
		// working set to restrict retrieval
		// UNUSED
		std::vector<std::string> workingSet;
		std::vector<std::string> relFBDocs;

		q.push( new query_t( i, queryNumber, queryText, queryType, workingSet, relFBDocs ) );

	}
}



/**
 * Main method, entry point of the application
 */
int main(int argc, char * argv[])
{
	if(argc<2){
		std::cerr<<"[ERROR] At least one parameter file needs to be provided"<<std::endl;
		return 0;
	}

	std::cout<<"Parsing parameter file "<<argv[1]<<" ..."<<std::endl;
	indri::api::Parameters& param = indri::api::Parameters::instance();
	param.loadFile(argv[1]);

	int cutAtDocRank = -1;
	if(argc==3){
		cutAtDocRank = atoi(argv[2]);
		std::cout<<"[INFO] Employing a document rank cutoff of "<<cutAtDocRank<<std::endl;
	}


	//open the index (parameter <index>)
	//provides easy access to the TermList per document
	indri::collection::Repository repository;
	indri::index::Index *thisIndex;

	//provides easy access to compute retrieval status values
	indri::api::QueryEnvironment *queryEnv    = new indri::api::QueryEnvironment();
	indri::api::QueryEnvironment *queryEnvJM  = new indri::api::QueryEnvironment();
	indri::api::QueryEnvironment *queryEnvDIR = new indri::api::QueryEnvironment();
	indri::api::QueryEnvironment *queryEnvTS  = new indri::api::QueryEnvironment();
	indri::api::QueryEnvironment *queryEnvTFIDF=new indri::api::QueryEnvironment();
	indri::api::QueryEnvironment *queryEnvOkapi=new indri::api::QueryEnvironment();

	try
	{
		std::cout<<"Opening index at "<<param.get("index","")<<std::endl;
		repository.openRead(param.get("index",""));
		indri::collection::Repository::index_state repIndexState = repository.indexes();
    		thisIndex=(*repIndexState)[0];

		queryEnv->addIndex(param.get("index",""));
		queryEnvJM->addIndex(param.get("index",""));
		queryEnvDIR->addIndex(param.get("index",""));
		queryEnvTS->addIndex(param.get("index",""));
		queryEnvTFIDF->addIndex(param.get("index",""));
		queryEnvOkapi->addIndex(param.get("index",""));

		std::vector<std::string> rulesJM;
		rulesJM.push_back("method:linear");
		rulesJM.push_back("lambda:0.5");
		queryEnvJM->setScoringRules(rulesJM);

		std::vector<std::string> rulesDIR;
		rulesDIR.push_back("method:dirichlet");
		rulesDIR.push_back("mu:1500");
		queryEnvDIR->setScoringRules(rulesDIR);

		std::vector<std::string> rulesTS;
		rulesTS.push_back("method:dirichlet");
		rulesTS.push_back("mu:100");
		queryEnvTS->setScoringRules(rulesTS);

		queryEnvTFIDF->setBaseline("tfidf,k1:1.2,b:0.75");

		queryEnvOkapi->setBaseline("okapi,k1:1.2,b:0.75,k3:7");
	}
	catch (Exception &ex)
	{
		std::cerr<<"[ERROR] Something went wrong when opening the IndexEnv and QueryEnv!"<<std::endl;
		return 0;
	}


	//open final result file (=feature file in RankLib format)
	std::cout<<"Opening outfile stream"<<std::endl;
	std::ofstream outFile(param.get("outFile","").c_str());
	if(!outFile.is_open()){
		std::cerr<<"[ERROR] Unable to open the output file for writing!"<<std::endl;
		return 0;
	}
	//read the queries from file
	std::queue< query_t* > queries;
    	indri::api::Parameters parameterQueries = param[ "query" ];
	int queryOffset = param.get( "queryOffset", 0 );
	push_queue( queries, parameterQueries, queryOffset );
    	int queryCount = (int)queries.size();
    	std::cout<<"Number of queries read: "<<queryCount<<std::endl;
	int counter = 0;

	//read the qrels from file (unranked = non-relevant)
	std::ifstream qrels(param.get("qrelsFile","").c_str());
	if(!qrels.is_open()){
		std::cerr<<"[ERROR] Unable to open the qrels file for reding"<<std::endl;
		return 0;
	}

	std::string qid, dummy, docid, rel;
	std::map<std::string,std::string> qrelsMap;
	while(qrels >> qid >> dummy >> docid >> rel)
	{
		qid.append("-").append(docid);//combined key
		qrelsMap.insert( std::make_pair(qid,rel) );
	}
	std::cout<<"Number qrels lines read: "<<qrelsMap.size()<<std::endl;
	qrels.close();


	indri::parse::Porter_Stemmer * stemmer = new indri::parse::Porter_Stemmer();
	std::cout<<"WARNING: assuming Porter stemming!"<<std::endl;
	std::cout<<"WARNING II: assuming that queries are _NOT_ already stemmed!"<<std::endl;

	//process the queries: lowercasing and Porter stemming (hardcoded)
	while(!queries.empty())
	{
		//time how long each query takes to process
		std::clock_t beginClock = clock();

		counter++;
		query_t* query = queries.front();


		//move from query string to query-term vector
		std:vector<std::string> queryTokens;

		std::string normalizedQueryText;
		std::string buf;
		stringstream ss(query->text);
		while(ss>>buf){

			std::cout<<"[normalization check] "<<buf<<"-->";

			//lowercase
			std::transform(buf.begin(), buf.end(), buf.begin(), ::tolower);

			//remove non-alphanumeric
			buf.erase(std::remove_if(buf.begin(), buf.end(), isNotAlpha ), buf.end());

			//Porter stemming
			char *word = (char*)buf.c_str();
			int ret = stemmer->porter_stem(word, 0, strlen(word) -1);
			word[ret+1] = 0;

			std::cout<<"["<<word<<"]"<<std::endl;

			//does the token appear in the index?
			if(queryEnvJM->documentCount(word)>0)
			{
				queryTokens.push_back(word);
				if(normalizedQueryText.length()>0){
					normalizedQueryText.append(" ");
				}
				normalizedQueryText.append(word);
			}
			else {
				std::cerr<<"[WARNING] Stemmed term "<<word<<" was _not_ found in the index!"<<std::endl;
			}
		}
		std::cout<<"Parsed query: ["<<normalizedQueryText<<"]"<<std::endl;
                if(normalizedQueryText.empty()){
                    std::cerr<<"[WARNING] Could not parse any term on query "<<query->text<<std::endl;
                    queries.pop();
                    continue;
                }

		//remove from vector queue
		queries.pop();

		//walk over the ranked list of documents
		//every time we hit a document that was ranked for the query we generate features
		std::ifstream documents(param.get("rankedDocsFile","").c_str());
		if(!documents.is_open()){
			std::cerr<<"[ERROR] Unable to open the document result file for reading!"<<std::endl;
			return 0;
		}

		bool queryDocsFound = false;
		int rank;
		std::string score, dummy2;//we also need qid, dummy, docid, rel (defined earlier)
		while(documents >> qid >> dummy >> docid >> rank >> score >> dummy2)
		{
			//right query?
			if(query->number.compare(qid)!=0 && queryDocsFound==false)
				continue;
			//we assume a sorted result file (all docs scored for a query follow each other)
			else if(query->number.compare(qid)!=0 && queryDocsFound==true)
				break; //we already saw a bunch of documents for that query, break
			else
				queryDocsFound = true;

			//above the document cutoff?
			if(cutAtDocRank>-1 && rank>cutAtDocRank)
				break;

			std::cout<<"\tquery: "<<qid<<", document rank: "<<rank<<std::endl;

			//find the relevance label in our qrels
			std::string combinedKey = qid;
			combinedKey.append("-").append(docid);
			std::string relevanceLabel = "0"; //by default non-relevant
			if(qrelsMap.find(combinedKey)!=qrelsMap.end())
			{
				relevanceLabel = qrelsMap.find(combinedKey)->second;
			}

			//convert external document identifier to an internal one
			std::vector<std::string> externalDocVec;
			externalDocVec.push_back(docid);
			//a vector containing a single docid
			std::vector<int> internalDocVec = queryEnv->documentIDsFromMetadata("docno",externalDocVec);

			std::vector<indri::api::ScoredExtentResult> resVec;

			//feature 1: LMIR.JM
			try {
				resVec = queryEnvJM->runQuery("#combine("+query->text+")",internalDocVec,1);
			}
			catch (Exception &ex)
			{
				std::cerr<<"[ERROR] queryEnvJM caused a problem!"<<std::endl;
				return 0;
			}
			double f1 = resVec[0].score;

			//feature 2: LMIR.DIR
			resVec.clear();
			try {
				resVec = queryEnvDIR->runQuery("#combine("+query->text+")",internalDocVec,1);
			}
			catch(Exception &ex)
			{
				std::cerr<<"[ERROR] queryEnvDIR caused a problem!"<<std::endl;
				return 0;
			}
			double f2 = resVec[0].score;

			//feature 3: LM.TWOSTAGE
			resVec.clear();
			try {
				resVec = queryEnvTS->runQuery("#combine("+query->text+")",internalDocVec,1);
			}
			catch(Exception &ex)
			{
				std::cerr<<"[ERROR] queryEnvTS caused a problem!"<<std::endl;
				return 0;
			}
			double f3 = resVec[0].score;

			//feature 4: TF.IDF with BM25 weighting
			resVec.clear();
			try {
				resVec = queryEnvTFIDF->runQuery(normalizedQueryText,internalDocVec,1);
			}
			catch(Exception &ex)
			{
				std::cerr<<"[ERROR] queryEnvTFIDF caused a problem!"<<std::endl;
				ex.writeMessage();
				return 0;
			}
			double f4 = resVec[0].score;

			//feature 5: Okapi
			resVec.clear();
			try {
				resVec = queryEnvOkapi->runQuery(normalizedQueryText,internalDocVec,1);
			}
			catch(Exception &ex)
			{
				std::cerr<<"[ERROR] queryEnvOkapi caused a problem!"<<std::endl;
				return 0;
			}
			double f5 = resVec[0].score;

			//compute document tf's
			std::map<std::string,int> tfMap;
			const indri::index::TermList *termList=thisIndex->termList(internalDocVec[0]);
			if (termList)
			{
				indri::utility::greedy_vector<lemur::api::TERMID_T > terms = termList->terms();
				for(int i=0; i<terms.size(); i++)
				{
					std::string term = thisIndex->term( termList->terms()[i]);

					std::map<std::string,int>::iterator it = tfMap.find(term);
					if( it != tfMap.end())
					{
						it->second++;
					}
					else
					{
						tfMap.insert(std::make_pair(term,1));
					}
				}
			}
			delete termList;

			double maxTF = -1000;
			for(std::map<std::string,int>::iterator it=tfMap.begin(); it!=tfMap.end(); ++it)
			{
				if( maxTF < it->second){
					maxTF = it->second;
				}
			}

			double minQueryTermTF = 1000;
			double maxQueryTermTF = 0;
			double sumQueryTermTF = 0;

			//tfidf-based features: compute for each query term the tf-idf
			std::vector<double> tfidfVec;
			std::vector<double> idfVec;
			int coveredQueryTermTokens = 0;
			int numQueryTerms = queryTokens.size();

            double idf_total = 0;
            double normIDF = 0;
            double maxIDF = -10000;
            double minIDF = 1000;
            double gamma1 = 0;
            double gamma2 = 0;
            double AvICTF = 0;
            double ICTF = 0;
            double SCS = 0;
            double query_length = queryTokens.size();
            double total_docs = 0;
			for(int i=0; i<queryTokens.size(); i++)
			{
				std::string queryToken = queryTokens[i];
				//idf
				double docsTotal = queryEnvJM->documentCount();
				double termTotal = queryEnvJM->termCount();
				double docsWithTerm = queryEnvJM->documentCount(queryToken);
				double TermWithTerm = queryEnvJM->termCount(queryToken);
//				cout<<"TermWithTerm = "<<TermWithTerm<<endl;
////				cout<<"docsWithTerm = "<<docsWithTerm<<endl;
//				cout<<"termTotal = "<<termTotal<<endl;

                idf_total += log(docsTotal/docsWithTerm);
                idfVec.push_back(log(docsTotal/docsWithTerm));
                ICTF += log(termTotal/TermWithTerm);

                if(maxIDF < log(docsTotal/docsWithTerm))
					maxIDF = log(docsTotal/docsWithTerm);
                if(minIDF > log(docsTotal/docsWithTerm))
					minIDF = log(docsTotal/docsWithTerm);

				if(docsWithTerm>0 && tfMap.find(queryToken)!=tfMap.end()){
					double idf = log(docsTotal/docsWithTerm);
					double tf = 0.5 + 0.5 * tfMap.find(queryToken)->second/maxTF;
//					idf_total += idf;
					tfidfVec.push_back( tf*idf );

					coveredQueryTermTokens++;

					if(minQueryTermTF > tf)
						minQueryTermTF = tf;
					if(maxQueryTermTF < tf)
						maxQueryTermTF = tf;
					sumQueryTermTF += tf;
				}
			}
            normIDF = idf_total / query_length;
            gamma2 = maxIDF / minIDF;

            // varIDF
            double varIDF = 0;
            for(int i=0; i<queryTokens.size(); i++)
			{
				varIDF += pow(idfVec[i]-normIDF,2);
			}
            varIDF = varIDF / queryTokens.size();
            gamma1 = sqrt(varIDF);
            AvICTF = ICTF / queryTokens.size();
            SCS = log(1/query_length) + AvICTF;

			double meanQueryTermTF = 0;
			if(coveredQueryTermTokens >0)
				meanQueryTermTF = sumQueryTermTF / coveredQueryTermTokens;

			//features based on TF.IDF
			double meanTFIDF = 0.0;
			double sumTFIDF = 0.0;
			double minTFIDF =  10000;
			double maxTFIDF = -10000;
			for(int i=0; i<tfidfVec.size(); i++)
			{
				meanTFIDF += tfidfVec[i];
				if(minTFIDF > tfidfVec[i])
					minTFIDF = tfidfVec[i];
				if(maxTFIDF < tfidfVec[i])
					maxTFIDF = tfidfVec[i];
			}
			sumTFIDF = meanTFIDF;

			if(tfidfVec.size()>0)
				meanTFIDF = meanTFIDF / tfidfVec.size();
			else
				meanTFIDF = 0;

			double varTFIDF = 0;
			for(int i=0; i<tfidfVec.size(); i++)
			{
				varTFIDF += pow(tfidfVec[i]-meanTFIDF,2);
			}

			if(tfidfVec.size()>0)
				varTFIDF = varTFIDF / tfidfVec.size();
			else
				varTFIDF = 0;

			//document length
			double doclen = queryEnvJM->documentLength(internalDocVec[0]);

			double normMinQueryTermTF = 0;
			double normMaxQueryTermTF = 0;
			double normSumQueryTermTF = 0;
			if(doclen > 0){
				normMinQueryTermTF = minQueryTermTF / doclen;
				normMaxQueryTermTF = maxQueryTermTF / doclen;
				normSumQueryTermTF = sumQueryTermTF / doclen;
			}

			double coveredQueryRatio = 0.0;
			if(numQueryTerms>0)
				coveredQueryRatio = coveredQueryTermTokens/numQueryTerms;

//			outFile<<relevanceLabel<<" qid:"<<qid<<" 1:"<<f1<<" 2:"<<f2<<" 3:"<<f3<<" 4:"<<f4<<" 5:"<<f5;
//			outFile<<" 6:"<<meanTFIDF<<" 7:"<<sumTFIDF<<" 8:"<<minTFIDF<<" 9:"<<maxTFIDF;
//			outFile<<" 10:"<<varTFIDF<<" 11:"<<doclen<<" 12:"<<coveredQueryTermTokens<<" 13:"<<coveredQueryRatio;
//			outFile<<" 14:"<<maxQueryTermTF<<" 15:"<<minQueryTermTF<<" 16:"<<sumQueryTermTF<<" 17:"<<meanQueryTermTF;
//			outFile<<" 18:"<<normMinQueryTermTF<<" 19:"<<normMaxQueryTermTF<<" 20:"<<normSumQueryTermTF<<" 21:"<<query_length<<" 22:"<<normIDF<<" 23:"<<maxIDF<<" 24:"<<gamma1<<" 25:"<<gamma2<<" 26:"<<AvICTF<<" 27:"<<SCS<<" #docid="<<docid<<std::endl;
//            outFile<<std::flush;

//			outFile<<relevanceLabel<<" qid:"<<qid<<" 1:"<<f1<<" 2:"<<f2<<" 3:"<<f3<<" 4:"<<f4<<" 5:"<<f5;
//			outFile<<" 6:"<<meanTFIDF<<" 7:"<<sumTFIDF<<" 8:"<<minTFIDF<<" 9:"<<maxTFIDF;
//			outFile<<" 10:"<<varTFIDF<<" 11:"<<coveredQueryTermTokens<<" 12:"<<coveredQueryRatio;
//			outFile<<" 13:"<<maxQueryTermTF<<" 14:"<<minQueryTermTF<<" 15:"<<sumQueryTermTF<<" 16:"<<meanQueryTermTF;
//			outFile<<" 17:"<<normMinQueryTermTF<<" 18:"<<normMaxQueryTermTF<<" 19:"<<normSumQueryTermTF<<" #docid="<<docid<<std::endl;
//			outFile<<std::flush;

//            outFile<<relevanceLabel<<" qid:"<<qid<<" 1:"<<idf_total<<" 2:"<<query_length<<" 3:"<<df<<" 4:"<<total_docs<<" #docid="<<docid<<std::endl;
//            outFile<<std::flush;


			outFile<<relevanceLabel<<" qid:"<<qid<<" 2:"<<f2<<" 5:"<<f5<<" #docid="<<docid<<std::endl;

			outFile<<std::flush;

		}
		documents.close();

		double durationInSeconds = ( std::clock() - beginClock ) / (double) CLOCKS_PER_SEC;
		std::cout<<"[TIMER] Processing query "<<query->number<<" took "<<durationInSeconds<<" seconds"<<std::endl;
	}
	outFile << std::flush;
	outFile.close();

	return 0;
}

