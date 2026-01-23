import React, { useState } from 'react';
import { Box, Typography, Button, Paper, TextField, Slider, Checkbox, FormControlLabel, CircularProgress, Alert, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip, IconButton } from '@mui/material';
import { Delete as DeleteIcon } from '@mui/icons-material';
import { api } from '../services/api';
import { useNavigate, useLocation } from 'react-router-dom';
import ModelSelectionDialog, { ModelType } from './ModelSelectionDialog';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import SaveIcon from '@mui/icons-material/Save';

// Helper to generate random stars
const generateStars = (count: number) => {
  return Array.from({ length: count }).map((_, i) => {
    const size = Math.random() * 2 + 1;
    const left = Math.random() * 100;
    const top = Math.random() * 100;
    const duration = 2 + Math.random() * 3;
    const delay = Math.random() * 5;
    return (
      <div
        key={i}
        className="twinkle-star"
        style={{
          position: 'absolute',
          left: `${left}%`,
          top: `${top}%`,
          width: size,
          height: size,
          borderRadius: '50%',
          background: 'white',
          opacity: 0.8,
          boxShadow: `0 0 ${size * 4}px 1px white`,
          animation: `twinkle ${duration}s infinite`,
          animationDelay: `${delay}s`,
          pointerEvents: 'none',
        }}
      />
    );
  });
};

const BatchPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // Get the selected model from navigation state, default to 'dash'
  const getInitialModel = (): ModelType => {
    if (location.state?.model) {
      return location.state.model;
    }
    const stored = localStorage.getItem('astrodash_selected_model');
    if (stored) {
      try {
        return JSON.parse(stored);
      } catch {
        return 'dash';
      }
    }
    return 'dash';
  };

  const [selectedModel, setSelectedModel] = useState<ModelType>(getInitialModel());
  const [modelDialogOpen, setModelDialogOpen] = useState(false);

  const [files, setFiles] = useState<File[]>([]);
  const [smoothing, setSmoothing] = useState<number>(0);
  const [knownZ, setKnownZ] = useState<boolean>(false);
  const [zValue, setZValue] = useState<string>('');
  const [minWave, setMinWave] = useState<string>('3500');
  const [maxWave, setMaxWave] = useState<string>('10000');
  const [calculateRlap, setCalculateRlap] = useState<boolean>(false);
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const newFiles = Array.from(event.target.files);
      setFiles(prevFiles => [...prevFiles, ...newFiles]);
    }
  };

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const downloadCSV = () => {
    if (!results) return;

    // Create CSV content
    const headers = [
      'File Name',
      'Status',
      'Best Match Type',
      'Best Match Age',
      'Probability',
      'RLAP',
      'Redshift'
    ];

    const csvContent = [
      headers.join(','),
      ...Object.entries(results).map(([fname, result]: any) => {
        const status = result.error ? 'Error' : 'Success';
        const type = result.classification?.best_match?.type ?? '';
        const age = result.classification?.best_match?.age ?? '';
        const probability = result.classification?.best_match?.probability?.toFixed(3) ?? '';
        const rlap = (selectedModel === 'dash' && calculateRlap &&
          result.classification?.best_match?.rlap !== undefined &&
          result.classification?.best_match?.rlap !== null &&
          result.classification?.best_match?.rlap !== "N/A")
          ? result.classification.best_match.rlap
          : '';
        const redshift = result.classification?.best_match?.redshift ?? '';

        return [
          `"${fname}"`,
          `"${status}"`,
          `"${type}"`,
          `"${age}"`,
          probability,
          rlap,
          redshift
        ].join(',');
      })
    ].join('\n');

    // Create and download file
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `batch_classification_results_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Handle model selection
  const handleModelSelect = (model: ModelType) => {
    // Update localStorage
    localStorage.setItem('astrodash_selected_model', JSON.stringify(model));
    // Update the selectedModel state
    setSelectedModel(model);
    setModelDialogOpen(false);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (files.length === 0) {
      setError('Please select at least one file to upload.');
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      // Handle model selection properly
      let modelType: 'dash' | 'transformer' | undefined = undefined;
      let model_id: string | undefined = undefined;
      if (selectedModel === 'dash' || selectedModel === 'transformer') {
        modelType = selectedModel;
      } else if (typeof selectedModel === 'object' && selectedModel.user) {
        model_id = selectedModel.user;
        // Explicitly set modelType to undefined when using user-uploaded model
        modelType = undefined;
      }

      const params = {
        smoothing,
        knownZ,
        zValue: knownZ ? parseFloat(zValue) : undefined,
        minWave: parseInt(minWave),
        maxWave: parseInt(maxWave),
        calculateRlap,
        ...(modelType ? { modelType } : {}),
        ...(model_id ? { model_id } : {}),
      };

      // Check if we have a zip file
      const zipFile = files.find(file => file.name.toLowerCase().endsWith('.zip'));

      if (zipFile && files.length === 1) {
        // Single zip file - use the original zip endpoint
        const response = await api.batchProcess({ zipFile, params });
        setResults(response);
      } else {
        // Multiple files or individual files - use the multiple files endpoint
        const response = await api.batchProcessMultiple({ files, params });
        setResults(response);
      }

      // Log the model type used
      console.log('Selected model:', selectedModel);
      console.log('Model type:', modelType);
      console.log('Model ID:', model_id);
      console.log(`Batch classification completed using ${typeof selectedModel === 'object' && selectedModel.user ? 'user-uploaded' : selectedModel} model`);

    } catch (err: any) {
      console.error('Batch processing error:', err);

      // Enhanced error handling for custom exceptions
      let errorMessage = 'Batch classification failed.';

      if (err.message) {
        errorMessage = err.message;
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.response?.data?.message) {
        errorMessage = err.response.data.message;
      }

      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', pt: 6, overflow: 'hidden', background: 'linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%)' }}>
      {/* Back button at top left */}
      <IconButton
        onClick={() => navigate(-1)}
        sx={{
          position: 'absolute',
          top: 24,
          left: 24,
          zIndex: 2,
          color: 'white',
          background: 'rgba(20, 30, 50, 0.7)',
          boxShadow: '0 2px 8px 0 rgba(0,0,0,0.25)',
          '&:hover': { background: 'rgba(20, 30, 50, 0.9)' },
        }}
        aria-label="Back"
        size="large"
      >
        <ArrowBackIcon fontSize="inherit" />
      </IconButton>
      {/* Twinkling stars background */}
      <style>{`
        @keyframes twinkle {
          0%, 100% { opacity: 0.8; }
          50% { opacity: 0.2; }
        }
        .twinkle-star {
          z-index: 0;
        }
      `}</style>
      <Box sx={{ position: 'absolute', inset: 0, width: '100%', height: '100%', zIndex: 0, pointerEvents: 'none' }}>
        {generateStars(80)}
      </Box>
      {/* Main content */}
      <Box sx={{ position: 'relative', zIndex: 1, width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <Typography variant="h3" gutterBottom sx={{ color: 'white', textShadow: '0 2px 8px #000' }}>
          Batch Classification
        </Typography>
          {/* Model indicator */}
          <Chip
            label={
              typeof selectedModel === 'object' && selectedModel.user
                ? 'User Model'
                : `${selectedModel === 'transformer' ? 'Transformer' : 'DASH'} Model`
            }
            onClick={() => setModelDialogOpen(true)}
            sx={{
              backgroundColor: typeof selectedModel === 'object' && selectedModel.user
                ? 'rgba(156, 39, 176, 0.2)'
                : selectedModel === 'transformer'
                  ? 'rgba(255, 152, 0, 0.2)'
                  : 'rgba(76, 175, 80, 0.2)',
              color: typeof selectedModel === 'object' && selectedModel.user
                ? '#9c27b0'
                : selectedModel === 'transformer'
                  ? '#ff9800'
                  : '#4caf50',
              border: `1px solid ${
                typeof selectedModel === 'object' && selectedModel.user
                  ? '#9c27b0'
                  : selectedModel === 'transformer'
                    ? '#ff9800'
                    : '#4caf50'
              }`,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              '&:hover': {
                transform: 'scale(1.05)',
                boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
              },
            }}
            size="small"
          />
        </Box>
        <Typography variant="body1" sx={{ mb: 4, color: 'white', textShadow: '0 1px 4px #000' }}>
          Upload multiple individual files or a zip file containing spectra for batch classification.
        </Typography>
        <Paper sx={{ p: 4, mb: 4, width: '100%', maxWidth: 600, background: 'rgba(20, 30, 50, 0.95)', boxShadow: '0 8px 32px 0 rgba(0,0,0,0.25)' }}>
          <form onSubmit={handleSubmit}>
            <Box sx={{ mb: 2 }}>
              <Button
                variant="contained"
                component="label"
                fullWidth
                sx={{ mb: 1 }}
              >
                Add Files
                <input
                  type="file"
                  accept=".fits,.txt,.dat,.csv,.zip"
                  multiple
                  hidden
                  onChange={handleFileChange}
                />
              </Button>
              {files.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" sx={{ color: 'white', mb: 1 }}>
                    Selected Files ({files.length}):
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {files.map((file, index) => (
                      <Chip
                        key={index}
                        label={file.name}
                        onDelete={() => removeFile(index)}
                        deleteIcon={<DeleteIcon />}
                        sx={{
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          color: 'white',
                          '& .MuiChip-deleteIcon': {
                            color: 'white'
                          }
                        }}
                      />
                    ))}
                  </Box>
                </Box>
              )}
            </Box>
            <Box sx={{ mb: 2 }}>
              <Typography gutterBottom sx={{ color: 'white' }}>Smoothing</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <TextField
                  size="small"
                  value={smoothing}
                  onChange={(e) => setSmoothing(Number(e.target.value))}
                  sx={{ width: 60, mr: 2,
                    '& .MuiInputBase-input': { color: 'white' },
                    '& .MuiOutlinedInput-root': {
                      '& fieldset': { borderColor: 'white' },
                      '&:hover fieldset': { borderColor: 'white' },
                      '&.Mui-focused fieldset': { borderColor: 'white' },
                    },
                  }}
                  type="number"
                  inputProps={{ min: 0, max: 50 }}
                  InputLabelProps={{ style: { color: 'white' } }}
                />
                <Slider
                  value={typeof smoothing === 'number' ? smoothing : 0}
                  onChange={(_event, newValue) => setSmoothing(newValue as number)}
                  aria-labelledby="smooth-slider"
                  valueLabelDisplay="auto"
                  step={1}
                  marks
                  min={0}
                  max={50}
                  sx={{ flex: 1 }}
                />
              </Box>
            </Box>
            <Box sx={{ mb: 2 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={knownZ}
                    onChange={(e) => setKnownZ(e.target.checked)}
                    sx={{ color: 'white' }}
                  />
                }
                label={<span style={{ color: 'white' }}>Known Redshift</span>}
              />
              {knownZ && (
                <TextField
                  size="small"
                  value={zValue}
                  onChange={(e) => setZValue(e.target.value)}
                  label="Redshift"
                  sx={{ ml: 2, width: 120,
                    '& .MuiInputBase-input': { color: 'white' },
                    '& .MuiOutlinedInput-root': {
                      '& fieldset': { borderColor: 'white' },
                      '&:hover fieldset': { borderColor: 'white' },
                      '&.Mui-focused fieldset': { borderColor: 'white' },
                    },
                  }}
                  type="number"
                  InputLabelProps={{ style: { color: 'white' } }}
                />
              )}
            </Box>
            {selectedModel === 'dash' && (
              <Box sx={{ mb: 2 }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={calculateRlap}
                      onChange={(e) => setCalculateRlap(e.target.checked)}
                      sx={{ color: 'white' }}
                    />
                  }
                  label={<span style={{ color: 'white' }}>Calculate RLAP</span>}
                />
              </Box>
            )}
            <Box sx={{ mb: 2, display: 'flex', gap: 2 }}>
              <TextField
                size="small"
                label="Min wave"
                value={minWave}
                onChange={(e) => setMinWave(e.target.value)}
                type="number"
                fullWidth
                sx={{
                  '& .MuiInputBase-input': { color: 'white' },
                  '& .MuiOutlinedInput-root': {
                    '& fieldset': { borderColor: 'white' },
                    '&:hover fieldset': { borderColor: 'white' },
                    '&.Mui-focused fieldset': { borderColor: 'white' },
                  },
                }}
                InputLabelProps={{ style: { color: 'white' } }}
              />
              <TextField
                size="small"
                label="Max wave"
                value={maxWave}
                onChange={(e) => setMaxWave(e.target.value)}
                type="number"
                fullWidth
                sx={{
                  '& .MuiInputBase-input': { color: 'white' },
                  '& .MuiOutlinedInput-root': {
                    '& fieldset': { borderColor: 'white' },
                    '&:hover fieldset': { borderColor: 'white' },
                    '&.Mui-focused fieldset': { borderColor: 'white' },
                  },
                }}
                InputLabelProps={{ style: { color: 'white' } }}
              />
            </Box>
            <Box sx={{ mt: 3 }}>
              <Button
                variant="contained"
                color="primary"
                type="submit"
                fullWidth
                disabled={loading}
                sx={{ py: 1.5 }}
              >
                {loading ? <CircularProgress size={24} color="inherit" /> : 'Run Batch Classification'}
              </Button>
            </Box>
          </form>
          {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
        </Paper>
        {results && (
          <Box sx={{ width: '90vw', maxWidth: 1200 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h5" sx={{ color: 'white', textShadow: '0 1px 4px #000' }}>Results</Typography>
              <Button
                variant="contained"
                color="primary"
                onClick={downloadCSV}
                disabled={loading}
                sx={{
                  minWidth: 'auto',
                  px: 2,
                  py: 1,
                  borderRadius: '50%',
                  width: 48,
                  height: 48
                }}
                title="Download CSV"
              >
                {loading ? <CircularProgress size={24} color="inherit" /> : <SaveIcon />}
              </Button>
            </Box>
            <TableContainer component={Paper} sx={{ mb: 4, background: 'rgba(20, 30, 50, 0.97)' }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>File Name</TableCell>
                    <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Status</TableCell>
                    <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Best Match Type</TableCell>
                    <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Best Match Age</TableCell>
                    <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Probability</TableCell>
                    <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>RLAP</TableCell>
                    <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Redshift</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(results).map(([fname, result]: any) => (
                    <TableRow key={fname}>
                      <TableCell sx={{ color: 'white' }}>{fname}</TableCell>
                      <TableCell>
                        {result.error ? (
                          <Alert severity="error">{result.error}</Alert>
                        ) : <span style={{ color: '#90ee90' }}>Success</span>}
                      </TableCell>
                      <TableCell sx={{ color: 'white' }}>{result.classification?.best_match?.type ?? '-'}</TableCell>
                      <TableCell sx={{ color: 'white' }}>{result.classification?.best_match?.age ?? '-'}</TableCell>
                      <TableCell sx={{ color: 'white' }}>{result.classification?.best_match?.probability?.toFixed(3) ?? '-'}</TableCell>
                      <TableCell sx={{ color: 'white' }}>{
                        selectedModel === 'dash' && calculateRlap &&
                          result.classification?.best_match?.rlap !== undefined &&
                          result.classification?.best_match?.rlap !== null &&
                          result.classification?.best_match?.rlap !== "N/A"
                          ? result.classification.best_match.rlap
                          : '-'
                      }</TableCell>
                      <TableCell sx={{ color: 'white' }}>{result.classification?.best_match?.redshift ?? '-'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </Box>

      {/* Model Selection Dialog */}
      <ModelSelectionDialog
        open={modelDialogOpen}
        onClose={() => setModelDialogOpen(false)}
        onModelSelect={handleModelSelect}
      />
    </Box>
  );
};

export default BatchPage;
