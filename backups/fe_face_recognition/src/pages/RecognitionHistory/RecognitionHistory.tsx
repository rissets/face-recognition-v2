import React, { useEffect, useState, useCallback } from 'react';
import axios from 'axios';

interface RecognitionRecord {
  id: number;
  person_name: string;
  confidence: number;
  timestamp: string;
  source: string;
  image_url?: string;
}

const RecognitionHistory: React.FC = () => {
  const [records, setRecords] = useState<RecognitionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [totalCount, setTotalCount] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');

  const fetchRecognitionHistory = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get('/recognition/history/', {
        params: {
          page: page + 1,
          page_size: rowsPerPage,
          search: searchTerm,
        },
      });
      
      setRecords(response.data.results);
      setTotalCount(response.data.count);
    } catch (error) {
      console.error('Error fetching recognition history:', error);
    } finally {
      setLoading(false);
    }
  }, [page, rowsPerPage, searchTerm]);

  useEffect(() => {
    fetchRecognitionHistory();
  }, [fetchRecognitionHistory]);

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
    setPage(0);
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-4xl font-bold glow-text mb-2">Recognition History</h1>
        <p className="text-gray-400">Track all face recognition activities</p>
      </div>

      <div className="cyber-card">
        <div className="p-6">
          <div className="mb-6">
            <div className="relative">
              <input
                type="text"
                placeholder="Search by person name..."
                value={searchTerm}
                onChange={handleSearchChange}
                className="cyber-input w-full pl-10"
              />
              <svg className="absolute left-3 top-3.5 h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
          </div>

          {loading ? (
            <div className="flex justify-center items-center min-h-[200px]">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400"></div>
            </div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-cyan-400/20">
                      <th className="text-left py-4 px-4 text-cyan-400 font-semibold">Person Name</th>
                      <th className="text-left py-4 px-4 text-cyan-400 font-semibold">Confidence</th>
                      <th className="text-left py-4 px-4 text-cyan-400 font-semibold">Timestamp</th>
                      <th className="text-left py-4 px-4 text-cyan-400 font-semibold">Source</th>
                      <th className="text-left py-4 px-4 text-cyan-400 font-semibold">Image</th>
                    </tr>
                  </thead>
                  <tbody>
                    {records.map((record, index) => (
                      <tr key={record.id} className={`border-b border-gray-700/50 hover:bg-cyan-400/5 transition-colors duration-200 ${index % 2 === 0 ? 'bg-gray-900/20' : ''}`}>
                        <td className="py-4 px-4">
                          <div className="font-medium text-white">
                            {record.person_name}
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            record.confidence >= 0.8 
                              ? 'bg-green-900/50 text-green-300 border border-green-500/50' 
                              : record.confidence >= 0.6 
                              ? 'bg-yellow-900/50 text-yellow-300 border border-yellow-500/50'
                              : 'bg-red-900/50 text-red-300 border border-red-500/50'
                          }`}>
                            {(record.confidence * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-4 px-4">
                          <div className="text-gray-300 text-sm">
                            {formatTimestamp(record.timestamp)}
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-800 text-gray-300 border border-gray-600">
                            {record.source}
                          </span>
                        </td>
                        <td className="py-4 px-4">
                          {record.image_url ? (
                            <img
                              src={record.image_url}
                              alt="Recognition"
                              className="w-15 h-11 object-cover rounded border border-cyan-400/30"
                            />
                          ) : (
                            <span className="text-gray-500 text-sm">No image</span>
                          )}
                        </td>
                      </tr>
                    ))}
                    {records.length === 0 && (
                      <tr>
                        <td colSpan={5} className="py-8 text-center">
                          <div className="text-gray-500">
                            <svg className="mx-auto h-12 w-12 text-gray-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            No recognition records found
                          </div>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              
              {/* Custom Pagination */}
              <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-700/50">
                <div className="text-sm text-gray-400">
                  Showing {page * rowsPerPage + 1} to {Math.min((page + 1) * rowsPerPage, totalCount)} of {totalCount} results
                </div>
                <div className="flex items-center space-x-2">
                  <label className="text-sm text-gray-400 mr-2">Rows per page:</label>
                  <select
                    value={rowsPerPage}
                    onChange={(e) => {
                      setRowsPerPage(parseInt(e.target.value));
                      setPage(0);
                    }}
                    className="bg-gray-800 border border-gray-600 text-white rounded px-2 py-1 text-sm focus:border-cyan-400 focus:outline-none"
                  >
                    {[5, 10, 25, 50].map(size => (
                      <option key={size} value={size}>{size}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => setPage(Math.max(0, page - 1))}
                    disabled={page === 0}
                    className="px-3 py-1 text-sm bg-gray-800 border border-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
                  >
                    Previous
                  </button>
                  <span className="text-sm text-gray-400">
                    Page {page + 1} of {Math.ceil(totalCount / rowsPerPage)}
                  </span>
                  <button
                    onClick={() => setPage(Math.min(Math.ceil(totalCount / rowsPerPage) - 1, page + 1))}
                    disabled={page >= Math.ceil(totalCount / rowsPerPage) - 1}
                    className="px-3 py-1 text-sm bg-gray-800 border border-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
                  >
                    Next
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecognitionHistory;