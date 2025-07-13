Inspector inspector = getInspector();
Future<Integer> sizeFuture = inspector.getSizeInBytes();
int inspectorSize = sizeFuture.get();
return Primitive.INT.sizeInBytes + inspectorSize;