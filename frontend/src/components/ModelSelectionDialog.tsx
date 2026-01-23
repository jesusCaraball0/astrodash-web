import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Card,
  CardContent,
  CardActionArea,
  Chip,
  TextField,
  IconButton,
  Tooltip,
  Dialog as EditDialog,
  DialogTitle as EditDialogTitle,
  DialogContent as EditDialogContent,
  DialogActions as EditDialogActions,
} from '@mui/material';
import { motion } from 'framer-motion';
import ScienceIcon from '@mui/icons-material/Science';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import { api } from '../services/api';

export type ModelType = 'dash' | 'transformer' | { user: string }; // user: model_id

interface ModelSelectionDialogProps {
  open: boolean;
  onClose: () => void;
  onModelSelect: (model: ModelType) => void;
}

const ModelSelectionDialog: React.FC<ModelSelectionDialogProps> = ({
  open,
  onClose,
  onModelSelect,
}) => {
  const [userModels, setUserModels] = React.useState<any[]>([]);
  const [uploading, setUploading] = React.useState(false);
  const [uploadError, setUploadError] = React.useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = React.useState<any | null>(null);
  const [file, setFile] = React.useState<File | null>(null);
  const [classMapping, setClassMapping] = React.useState('');
  const [inputShape, setInputShape] = React.useState('');
  const [modelName, setModelName] = React.useState('');
  const [modelDescription, setModelDescription] = React.useState('');
  const [editDialogOpen, setEditDialogOpen] = React.useState(false);
  const [editingModel, setEditingModel] = React.useState<any>(null);
  const [editName, setEditName] = React.useState('');
  const [editDescription, setEditDescription] = React.useState('');
  const [updating, setUpdating] = React.useState(false);

  React.useEffect(() => {
    // Fetch user models (stub for now)
    api.getUserModels().then(models => {
      console.log('Fetched user models:', models);
      setUserModels(models);
    });
  }, [uploadSuccess]);

  const handleModelSelect = (model: ModelType) => {
    console.log('ModelSelectionDialog: handleModelSelect called with:', model);
    onModelSelect(model);
    onClose();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);
    try {
      const parsedClassMapping = JSON.parse(classMapping);
      const parsedInputShape = JSON.parse(inputShape);
      const result = await api.uploadModel({
        file: file!,
        classMapping: parsedClassMapping,
        inputShape: parsedInputShape,
        name: modelName,
        description: modelDescription
      });
      setUploadSuccess(result);
      setFile(null);
      setClassMapping('');
      setInputShape('');
      setModelName('');
      setModelDescription('');
    } catch (err: any) {
      console.error('Model upload error:', err);

      // Enhanced error handling for custom exceptions
      let errorMessage = 'Upload failed';

      if (err.message) {
        errorMessage = err.message;
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.response?.data?.message) {
        errorMessage = err.response.data.message;
      }

      setUploadError(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteModel = async (modelId: string, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent card selection

    console.log('Attempting to delete model with ID:', modelId);
    console.log('Model ID type:', typeof modelId);

    if (!modelId || modelId === 'undefined') {
      alert('Invalid model ID. Cannot delete model.');
      return;
    }

    if (window.confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
      try {
        await api.deleteModel(modelId);
        // Refresh the user models list
        const updatedModels = await api.getUserModels();
        setUserModels(updatedModels);
      } catch (err: any) {
        console.error('Failed to delete model:', err);

        // Enhanced error handling for custom exceptions
        let errorMessage = 'Unknown error';

        if (err.message) {
          errorMessage = err.message;
        } else if (err.response?.data?.detail) {
          errorMessage = err.response.data.detail;
        } else if (err.response?.data?.message) {
          errorMessage = err.response.data.message;
        }

        alert('Failed to delete model: ' + errorMessage);
      }
    }
  };

  const handleEditModel = (model: any, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent card selection
    setEditingModel(model);
    setEditName(model.name || '');
    setEditDescription(model.description || '');
    setEditDialogOpen(true);
  };

  const handleUpdateModel = async () => {
    if (!editingModel) return;

    setUpdating(true);
    try {
      await api.updateModel(editingModel.model_id, {
        name: editName,
        description: editDescription
      });

      // Refresh the user models list
      const updatedModels = await api.getUserModels();
      setUserModels(updatedModels);

      setEditDialogOpen(false);
      setEditingModel(null);
      setEditName('');
      setEditDescription('');
    } catch (err: any) {
      console.error('Failed to update model:', err);

      // Enhanced error handling for custom exceptions
      let errorMessage = 'Unknown error';

      if (err.message) {
        errorMessage = err.message;
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.response?.data?.message) {
        errorMessage = err.response.data.message;
      }

      alert('Failed to update model: ' + errorMessage);
    } finally {
      setUpdating(false);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          background: 'linear-gradient(135deg, #1a237e 0%, #0d47a1 100%)',
          color: 'white',
          borderRadius: '16px',
        },
      }}
    >
      <DialogTitle sx={{ textAlign: 'center', pb: 1 }}>
        <Typography variant="h4" component="h2" sx={{ fontWeight: 600 }}>
          Choose Your Model
        </Typography>
        <Typography variant="body1" sx={{ opacity: 0.8, mt: 1 }}>
          Select the machine learning model for classification
        </Typography>
      </DialogTitle>

      <DialogContent sx={{ pt: 2 }}>
        <Box sx={{ display: 'flex', gap: 2, flexDirection: { xs: 'column', sm: 'row' } }}>
          {/* DASH Model Card - RECOMMENDED */}
          <motion.div
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            style={{ flex: 1 }}
          >
            <Card
              sx={{
                background: 'rgba(255, 255, 255, 0.1)',
                backdropFilter: 'blur(10px)',
                border: '2px solid rgba(76, 175, 80, 0.6)',
                color: 'white',
                height: '100%',
                boxShadow: '0 0 20px rgba(76, 175, 80, 0.4), 0 0 40px rgba(76, 175, 80, 0.2)',
                position: 'relative',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: -2,
                  left: -2,
                  right: -2,
                  bottom: -2,
                  background: 'linear-gradient(45deg, rgba(76, 175, 80, 0.3), rgba(129, 199, 132, 0.3))',
                  borderRadius: 'inherit',
                  zIndex: -1,
                  animation: 'pulse 2s infinite',
                },
                '@keyframes pulse': {
                  '0%': { opacity: 0.6 },
                  '50%': { opacity: 1 },
                  '100%': { opacity: 0.6 },
                },
              }}
            >
              <CardActionArea
                onClick={() => handleModelSelect('dash')}
                sx={{ height: '100%', p: 2 }}
              >
                <CardContent sx={{ textAlign: 'center' }}>
                  <Box sx={{ position: 'relative' }}>
                    <ScienceIcon sx={{ fontSize: 48, mb: 2, color: '#4caf50' }} />
                    <Chip
                      label="RECOMMENDED"
                      size="small"
                      sx={{
                        position: 'absolute',
                        top: -10,
                        right: -10,
                        backgroundColor: '#4caf50',
                        color: 'white',
                        fontWeight: 'bold',
                        fontSize: '0.7rem',
                        height: 20,
                      }}
                    />
                  </Box>
                  <Typography variant="h5" component="h3" sx={{ mb: 1, fontWeight: 600 }}>
                    DASH Model
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2, opacity: 0.8 }}>
                    CNN-based model with comprehensive template matching
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, justifyContent: 'center' }}>
                    <Chip
                      label="CNN"
                      size="small"
                      sx={{ backgroundColor: 'rgba(76, 175, 80, 0.2)', color: '#4caf50' }}
                    />
                    <Chip
                      label="Template Matching"
                      size="small"
                      sx={{ backgroundColor: 'rgba(76, 175, 80, 0.2)', color: '#4caf50' }}
                    />
                    <Chip
                      label="RLap Scores"
                      size="small"
                      sx={{ backgroundColor: 'rgba(76, 175, 80, 0.2)', color: '#4caf50' }}
                    />
                  </Box>
                </CardContent>
              </CardActionArea>
            </Card>
          </motion.div>

          {/* Transformer Model Card */}
          <motion.div
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            style={{ flex: 1 }}
          >
            <Card
              sx={{
                background: 'rgba(255, 255, 255, 0.1)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                color: 'white',
                height: '100%',
              }}
            >
              <CardActionArea
                onClick={() => handleModelSelect('transformer')}
                sx={{ height: '100%', p: 2 }}
              >
                <CardContent sx={{ textAlign: 'center' }}>
                  <AutoAwesomeIcon sx={{ fontSize: 48, mb: 2, color: '#ff9800' }} />
                  <Typography variant="h5" component="h3" sx={{ mb: 1, fontWeight: 600 }}>
                    Transformer Model
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2, opacity: 0.8 }}>
                    Advanced transformer-based model with 5-class classification
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, justifyContent: 'center' }}>
                    <Chip
                      label="Transformer"
                      size="small"
                      sx={{ backgroundColor: 'rgba(255, 152, 0, 0.2)', color: '#ff9800' }}
                    />
                    <Chip
                      label="5 Classes"
                      size="small"
                      sx={{ backgroundColor: 'rgba(255, 152, 0, 0.2)', color: '#ff9800' }}
                    />
                    <Chip
                      label="Fast Inference"
                      size="small"
                      sx={{ backgroundColor: 'rgba(255, 152, 0, 0.2)', color: '#ff9800' }}
                    />
                  </Box>
                </CardContent>
              </CardActionArea>
            </Card>
          </motion.div>
        </Box>

        {/* Upload New Model Card */}
        <Box sx={{ mt: 4 }}>
          <Card
            sx={{
              background: 'rgba(255, 255, 255, 0.1)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              color: 'white',
              minWidth: 280,
              maxWidth: 400,
              margin: '0 auto',
              mb: 3,
            }}
          >
            <CardContent sx={{ textAlign: 'center' }}>
              <CloudUploadIcon sx={{ fontSize: 48, mb: 2, color: '#2196f3' }} />
              <Typography variant="h5" component="h3" sx={{ mb: 1, fontWeight: 600 }}>
                Upload Your Model
              </Typography>
              <Typography variant="body2" sx={{ mb: 2, opacity: 0.8 }}>
                Upload a PyTorch .pth/.pt file, class mapping, and input shape
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, alignItems: 'center', mb: 1 }}>
                <Button
                  variant="contained"
                  component="label"
                  sx={{ mb: 1, background: '#2196f3', color: 'white' }}
                  startIcon={<CloudUploadIcon />}
                >
                  {file ? file.name : 'Choose Model File'}
                  <input
                    type="file"
                    accept=".pth,.pt"
                    hidden
                    onChange={handleFileChange}
                  />
                </Button>
                <TextField
                  label="Model Name"
                  value={modelName}
                  onChange={e => setModelName(e.target.value)}
                  size="small"
                  fullWidth
                  placeholder="Enter a name for your model"
                  sx={{ background: 'rgba(255,255,255,0.08)', borderRadius: 1, input: { color: 'white' }, label: { color: 'white' } }}
                />
                <TextField
                  label="Description"
                  value={modelDescription}
                  onChange={e => setModelDescription(e.target.value)}
                  size="small"
                  fullWidth
                  multiline
                  rows={2}
                  placeholder="Describe your model (optional)"
                  sx={{ background: 'rgba(255,255,255,0.08)', borderRadius: 1, input: { color: 'white' }, label: { color: 'white' } }}
                />
                <TextField
                  label="Class Mapping (JSON)"
                  value={classMapping}
                  onChange={e => setClassMapping(e.target.value)}
                  size="small"
                  fullWidth
                  multiline
                  minRows={2}
                  placeholder='{"Ia": 0, "Ib": 1, "Ic": 2}'
                  sx={{ background: 'rgba(255,255,255,0.08)', borderRadius: 1, input: { color: 'white' }, label: { color: 'white' } }}
                />
                <TextField
                  label="Input Shape (JSON, e.g. [1,1024])"
                  value={inputShape}
                  onChange={e => setInputShape(e.target.value)}
                  size="small"
                  fullWidth
                  placeholder="[1, 1024] or [[1, 1024], [1, 1024], [1, 1]]"
                  sx={{ background: 'rgba(255,255,255,0.08)', borderRadius: 1, input: { color: 'white' }, label: { color: 'white' } }}
                />
                <Button
                  variant="outlined"
                  color="primary"
                  onClick={handleUpload}
                  disabled={!file || !classMapping || !inputShape || !modelName || uploading}
                  sx={{ mt: 1 }}
                >
                  {uploading ? 'Uploading...' : 'Upload Model'}
                </Button>
                {uploadError && <Typography color="error" sx={{ mt: 1 }}>{uploadError}</Typography>}
                {uploadSuccess && (
                  <Box sx={{ mt: 1, color: '#90ee90', fontSize: 14 }}>
                    <Box sx={{ mb: 2 }}>
                      Uploaded! Model ID: <b>{uploadSuccess.model_id}</b>
                      <br />Output shape: {JSON.stringify(uploadSuccess.output_shape)}
                      <br />Input shape: {JSON.stringify(uploadSuccess.input_shape)}
                    </Box>
                    <Button
                      variant="contained"
                      color="success"
                      size="small"
                      onClick={() => handleModelSelect({ user: uploadSuccess.model_id })}
                    >
                      Use This Model
                    </Button>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Box>

        {/* User-uploaded Models */}
        {userModels.length > 0 && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" sx={{ mb: 2, color: 'white', fontWeight: 500 }}>
              User-Uploaded Models
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              {userModels
                .sort((a, b) => {
                  // Sort by creation date, newest first
                  const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
                  const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
                  return dateB - dateA;
                })
                .map((model) => (
                <Card
                  key={model.model_id}
                  sx={{
                    background: 'rgba(255,255,255,0.08)',
                    color: 'white',
                    minWidth: 260,
                    maxWidth: 320,
                    position: 'relative',
                  }}
                >
                  <CardActionArea onClick={() => {
                    console.log('Card clicked for model:', model);
                    handleModelSelect({ user: model.model_id });
                  }}>
                    <CardContent>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                        {model.name || model.model_filename || `Model ID: ${model.model_id}`}
                      </Typography>
                      {model.description && (
                        <Typography variant="body2" sx={{ opacity: 0.8, mb: 1 }}>
                          {model.description}
                        </Typography>
                      )}
                      <Typography variant="body2" sx={{ opacity: 0.8, mb: 1 }}>
                        Filename: {model.model_filename}
                      </Typography>
                      <Typography variant="body2" sx={{ opacity: 0.8, mb: 1 }}>
                        Input Shape: {model.input_shape ?
                          (Array.isArray(model.input_shape[0]) ?
                            model.input_shape.map((shape: any, i: number) => `Input ${i+1}: [${shape.join(', ')}]`).join(', ') :
                            `[${model.input_shape.join(', ')}]`
                          ) : 'N/A'}
                      </Typography>
                      <Typography variant="body2" sx={{ opacity: 0.8, mb: 1 }}>
                        Classes: {model.class_mapping ? Object.keys(model.class_mapping).join(', ') : 'N/A'}
                      </Typography>
                    </CardContent>
                  </CardActionArea>

                  {/* Action buttons */}
                  <Box sx={{
                    position: 'absolute',
                    top: 8,
                    right: 8,
                    display: 'flex',
                    gap: 0.5,
                    zIndex: 1
                  }}>
                    <Tooltip title="Edit model">
                      <IconButton
                        size="small"
                        onClick={(e) => handleEditModel(model, e)}
                        sx={{
                          color: 'rgba(255, 255, 255, 0.7)',
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '&:hover': {
                            color: 'white',
                            backgroundColor: 'rgba(255, 255, 255, 0.2)',
                          },
                        }}
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete model">
                      <IconButton
                        size="small"
                        onClick={(e) => handleDeleteModel(model.id || model.model_id, e)}
                        sx={{
                          color: 'rgba(255, 255, 255, 0.7)',
                          backgroundColor: 'rgba(255, 0, 0, 0.1)',
                          '&:hover': {
                            color: 'white',
                            backgroundColor: 'rgba(255, 0, 0, 0.3)',
                          },
                        }}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Card>
              ))}
            </Box>
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{ p: 3, pt: 1 }}>
        <Button
          onClick={onClose}
          sx={{
            color: 'white',
            borderColor: 'rgba(255, 255, 255, 0.3)',
            '&:hover': {
              borderColor: 'white',
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
            },
          }}
        >
          Cancel
        </Button>
      </DialogActions>

      {/* Edit Model Dialog */}
      <EditDialog
        open={editDialogOpen}
        onClose={() => setEditDialogOpen(false)}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            background: 'linear-gradient(135deg, #1a237e 0%, #0d47a1 100%)',
            color: 'white',
            borderRadius: '16px',
          },
        }}
      >
        <EditDialogTitle sx={{ textAlign: 'center' }}>
          <Typography variant="h5" component="h2" sx={{ fontWeight: 600 }}>
            Edit Model
          </Typography>
        </EditDialogTitle>
        <EditDialogContent sx={{ pt: 2 }}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="Model Name"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              fullWidth
              sx={{
                '& .MuiOutlinedInput-root': {
                  '& fieldset': { borderColor: 'rgba(255, 255, 255, 0.3)' },
                  '&:hover fieldset': { borderColor: 'rgba(255, 255, 255, 0.5)' },
                  '&.Mui-focused fieldset': { borderColor: 'white' },
                },
                '& .MuiInputLabel-root': { color: 'rgba(255, 255, 255, 0.7)' },
                '& .MuiInputBase-input': { color: 'white' },
              }}
            />
            <TextField
              label="Description"
              value={editDescription}
              onChange={(e) => setEditDescription(e.target.value)}
              fullWidth
              multiline
              rows={3}
              sx={{
                '& .MuiOutlinedInput-root': {
                  '& fieldset': { borderColor: 'rgba(255, 255, 255, 0.3)' },
                  '&:hover fieldset': { borderColor: 'rgba(255, 255, 255, 0.5)' },
                  '&.Mui-focused fieldset': { borderColor: 'white' },
                },
                '& .MuiInputLabel-root': { color: 'rgba(255, 255, 255, 0.7)' },
                '& .MuiInputBase-input': { color: 'white' },
              }}
            />
          </Box>
        </EditDialogContent>
        <EditDialogActions sx={{ p: 3, pt: 1 }}>
          <Button
            onClick={() => setEditDialogOpen(false)}
            sx={{
              color: 'white',
              borderColor: 'rgba(255, 255, 255, 0.3)',
              '&:hover': {
                borderColor: 'white',
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
              },
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleUpdateModel}
            disabled={updating}
            variant="contained"
            sx={{
              backgroundColor: '#2196f3',
              color: 'white',
              '&:hover': {
                backgroundColor: '#1976d2',
              },
            }}
          >
            {updating ? 'Updating...' : 'Update Model'}
          </Button>
        </EditDialogActions>
      </EditDialog>
    </Dialog>
  );
};

export default ModelSelectionDialog;
