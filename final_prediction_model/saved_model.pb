эА1
ч=║=
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
ю
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeintИ
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	
Р
)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
:
Less
x"T
y"T
z
"
Ttype:
2	
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
$

LogicalAnd
x

y

z
Р
!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	Р
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
М
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	Р
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
К
ReverseSequence

input"T
seq_lengths"Tlen
output"T"
seq_dimint"
	batch_dimint "	
Ttype"
Tlentype0	:
2	
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
s
	ScatterNd
indices"Tindices
updates"T
shape"Tindices
output"T"	
Ttype"
Tindicestype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
A

StackPopV2

handle
elem"	elem_type"
	elem_typetypeИ
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( И
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring И
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:И
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestringИ
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetypeИ
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
TtypeИ
9
TensorArraySizeV3

handle
flow_in
sizeИ
▐
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring И
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
TtypeИ
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.11.02v1.11.0-rc2-4-gc19e29306cєФ+
П
inputs/PlaceholderPlaceholder*
dtype0*4
_output_shapes"
 :                  '*)
shape :                  '
w
inputs/Placeholder_1Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
Y
inputs/Placeholder_2Placeholder*
dtype0*
_output_shapes
:*
shape:
e
 blstm_0/DropoutWrapperInit/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
g
"blstm_0/DropoutWrapperInit/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"blstm_0/DropoutWrapperInit/Const_2Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"blstm_0/DropoutWrapperInit_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
i
$blstm_0/DropoutWrapperInit_1/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
i
$blstm_0/DropoutWrapperInit_1/Const_2Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
$blstm_0/bidirectional_rnn/fw/fw/RankConst*
value	B :*
dtype0*
_output_shapes
: 
m
+blstm_0/bidirectional_rnn/fw/fw/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
m
+blstm_0/bidirectional_rnn/fw/fw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
╓
%blstm_0/bidirectional_rnn/fw/fw/rangeRange+blstm_0/bidirectional_rnn/fw/fw/range/start$blstm_0/bidirectional_rnn/fw/fw/Rank+blstm_0/bidirectional_rnn/fw/fw/range/delta*
_output_shapes
:*

Tidx0
А
/blstm_0/bidirectional_rnn/fw/fw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
m
+blstm_0/bidirectional_rnn/fw/fw/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ё
&blstm_0/bidirectional_rnn/fw/fw/concatConcatV2/blstm_0/bidirectional_rnn/fw/fw/concat/values_0%blstm_0/bidirectional_rnn/fw/fw/range+blstm_0/bidirectional_rnn/fw/fw/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
╛
)blstm_0/bidirectional_rnn/fw/fw/transpose	Transposeinputs/Placeholder&blstm_0/bidirectional_rnn/fw/fw/concat*4
_output_shapes"
 :                  '*
Tperm0*
T0
t
/blstm_0/bidirectional_rnn/fw/fw/sequence_lengthIdentityinputs/Placeholder_2*
_output_shapes
:*
T0
О
%blstm_0/bidirectional_rnn/fw/fw/ShapeShape)blstm_0/bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:
}
3blstm_0/bidirectional_rnn/fw/fw/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:

5blstm_0/bidirectional_rnn/fw/fw/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5blstm_0/bidirectional_rnn/fw/fw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Щ
-blstm_0/bidirectional_rnn/fw/fw/strided_sliceStridedSlice%blstm_0/bidirectional_rnn/fw/fw/Shape3blstm_0/bidirectional_rnn/fw/fw/strided_slice/stack5blstm_0/bidirectional_rnn/fw/fw/strided_slice/stack_15blstm_0/bidirectional_rnn/fw/fw/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
Ъ
Xblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ь
Tblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims
ExpandDims-blstm_0/bidirectional_rnn/fw/fw/strided_sliceXblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
Ъ
Oblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ConstConst*
valueB:А*
dtype0*
_output_shapes
:
Ч
Ublstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ф
Pblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concatConcatV2Tblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDimsOblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ConstUblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ъ
Ublstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
┼
Oblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zerosFillPblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concatUblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros/Const*
T0*

index_type0*(
_output_shapes
:         А
Ь
Zblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
а
Vblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1
ExpandDims-blstm_0/bidirectional_rnn/fw/fw/strided_sliceZblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dim*
_output_shapes
:*

Tdim0*
T0
Ь
Qblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Const*
valueB:А*
dtype0*
_output_shapes
:
Ь
Zblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
а
Vblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2
ExpandDims-blstm_0/bidirectional_rnn/fw/fw/strided_sliceZblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dim*
T0*
_output_shapes
:*

Tdim0
Ь
Qblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Const*
valueB:А*
dtype0*
_output_shapes
:
Щ
Wblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ь
Rblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2Vblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2Qblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Wblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
Ь
Wblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╦
Qblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1FillRblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1Wblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/Const*(
_output_shapes
:         А*
T0*

index_type0
Ь
Zblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dimConst*
dtype0*
_output_shapes
: *
value	B : 
а
Vblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3
ExpandDims-blstm_0/bidirectional_rnn/fw/fw/strided_sliceZblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dim*
T0*
_output_shapes
:*

Tdim0
Ь
Qblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/Const_3Const*
valueB:А*
dtype0*
_output_shapes
:
Я
'blstm_0/bidirectional_rnn/fw/fw/Shape_1Shape/blstm_0/bidirectional_rnn/fw/fw/sequence_length*
T0*
out_type0*#
_output_shapes
:         
Ц
%blstm_0/bidirectional_rnn/fw/fw/stackPack-blstm_0/bidirectional_rnn/fw/fw/strided_slice*
T0*

axis *
N*
_output_shapes
:
м
%blstm_0/bidirectional_rnn/fw/fw/EqualEqual'blstm_0/bidirectional_rnn/fw/fw/Shape_1%blstm_0/bidirectional_rnn/fw/fw/stack*
T0*#
_output_shapes
:         
o
%blstm_0/bidirectional_rnn/fw/fw/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
н
#blstm_0/bidirectional_rnn/fw/fw/AllAll%blstm_0/bidirectional_rnn/fw/fw/Equal%blstm_0/bidirectional_rnn/fw/fw/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
╝
,blstm_0/bidirectional_rnn/fw/fw/Assert/ConstConst*
dtype0*
_output_shapes
: *`
valueWBU BOExpected shape for Tensor blstm_0/bidirectional_rnn/fw/fw/sequence_length:0 is 

.blstm_0/bidirectional_rnn/fw/fw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
─
4blstm_0/bidirectional_rnn/fw/fw/Assert/Assert/data_0Const*`
valueWBU BOExpected shape for Tensor blstm_0/bidirectional_rnn/fw/fw/sequence_length:0 is *
dtype0*
_output_shapes
: 
Е
4blstm_0/bidirectional_rnn/fw/fw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
╕
-blstm_0/bidirectional_rnn/fw/fw/Assert/AssertAssert#blstm_0/bidirectional_rnn/fw/fw/All4blstm_0/bidirectional_rnn/fw/fw/Assert/Assert/data_0%blstm_0/bidirectional_rnn/fw/fw/stack4blstm_0/bidirectional_rnn/fw/fw/Assert/Assert/data_2'blstm_0/bidirectional_rnn/fw/fw/Shape_1*
T
2*
	summarize
╗
+blstm_0/bidirectional_rnn/fw/fw/CheckSeqLenIdentity/blstm_0/bidirectional_rnn/fw/fw/sequence_length.^blstm_0/bidirectional_rnn/fw/fw/Assert/Assert*
T0*
_output_shapes
:
Р
'blstm_0/bidirectional_rnn/fw/fw/Shape_2Shape)blstm_0/bidirectional_rnn/fw/fw/transpose*
_output_shapes
:*
T0*
out_type0

5blstm_0/bidirectional_rnn/fw/fw/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
Б
7blstm_0/bidirectional_rnn/fw/fw/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Б
7blstm_0/bidirectional_rnn/fw/fw/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
г
/blstm_0/bidirectional_rnn/fw/fw/strided_slice_1StridedSlice'blstm_0/bidirectional_rnn/fw/fw/Shape_25blstm_0/bidirectional_rnn/fw/fw/strided_slice_1/stack7blstm_0/bidirectional_rnn/fw/fw/strided_slice_1/stack_17blstm_0/bidirectional_rnn/fw/fw/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Р
'blstm_0/bidirectional_rnn/fw/fw/Shape_3Shape)blstm_0/bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:

5blstm_0/bidirectional_rnn/fw/fw/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
Б
7blstm_0/bidirectional_rnn/fw/fw/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Б
7blstm_0/bidirectional_rnn/fw/fw/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
г
/blstm_0/bidirectional_rnn/fw/fw/strided_slice_2StridedSlice'blstm_0/bidirectional_rnn/fw/fw/Shape_35blstm_0/bidirectional_rnn/fw/fw/strided_slice_2/stack7blstm_0/bidirectional_rnn/fw/fw/strided_slice_2/stack_17blstm_0/bidirectional_rnn/fw/fw/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
p
.blstm_0/bidirectional_rnn/fw/fw/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
╩
*blstm_0/bidirectional_rnn/fw/fw/ExpandDims
ExpandDims/blstm_0/bidirectional_rnn/fw/fw/strided_slice_2.blstm_0/bidirectional_rnn/fw/fw/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
r
'blstm_0/bidirectional_rnn/fw/fw/Const_1Const*
dtype0*
_output_shapes
:*
valueB:А
o
-blstm_0/bidirectional_rnn/fw/fw/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Є
(blstm_0/bidirectional_rnn/fw/fw/concat_1ConcatV2*blstm_0/bidirectional_rnn/fw/fw/ExpandDims'blstm_0/bidirectional_rnn/fw/fw/Const_1-blstm_0/bidirectional_rnn/fw/fw/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
p
+blstm_0/bidirectional_rnn/fw/fw/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
╔
%blstm_0/bidirectional_rnn/fw/fw/zerosFill(blstm_0/bidirectional_rnn/fw/fw/concat_1+blstm_0/bidirectional_rnn/fw/fw/zeros/Const*(
_output_shapes
:         А*
T0*

index_type0
|
&blstm_0/bidirectional_rnn/fw/fw/Rank_1Rank+blstm_0/bidirectional_rnn/fw/fw/CheckSeqLen*
T0*
_output_shapes
: 
o
-blstm_0/bidirectional_rnn/fw/fw/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
o
-blstm_0/bidirectional_rnn/fw/fw/range_1/deltaConst*
dtype0*
_output_shapes
: *
value	B :
ч
'blstm_0/bidirectional_rnn/fw/fw/range_1Range-blstm_0/bidirectional_rnn/fw/fw/range_1/start&blstm_0/bidirectional_rnn/fw/fw/Rank_1-blstm_0/bidirectional_rnn/fw/fw/range_1/delta*#
_output_shapes
:         *

Tidx0
╛
#blstm_0/bidirectional_rnn/fw/fw/MinMin+blstm_0/bidirectional_rnn/fw/fw/CheckSeqLen'blstm_0/bidirectional_rnn/fw/fw/range_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
|
&blstm_0/bidirectional_rnn/fw/fw/Rank_2Rank+blstm_0/bidirectional_rnn/fw/fw/CheckSeqLen*
T0*
_output_shapes
: 
o
-blstm_0/bidirectional_rnn/fw/fw/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
o
-blstm_0/bidirectional_rnn/fw/fw/range_2/deltaConst*
dtype0*
_output_shapes
: *
value	B :
ч
'blstm_0/bidirectional_rnn/fw/fw/range_2Range-blstm_0/bidirectional_rnn/fw/fw/range_2/start&blstm_0/bidirectional_rnn/fw/fw/Rank_2-blstm_0/bidirectional_rnn/fw/fw/range_2/delta*#
_output_shapes
:         *

Tidx0
╛
#blstm_0/bidirectional_rnn/fw/fw/MaxMax+blstm_0/bidirectional_rnn/fw/fw/CheckSeqLen'blstm_0/bidirectional_rnn/fw/fw/range_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
f
$blstm_0/bidirectional_rnn/fw/fw/timeConst*
dtype0*
_output_shapes
: *
value	B : 
╪
+blstm_0/bidirectional_rnn/fw/fw/TensorArrayTensorArrayV3/blstm_0/bidirectional_rnn/fw/fw/strided_slice_1*K
tensor_array_name64blstm_0/bidirectional_rnn/fw/fw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *%
element_shape:         А*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
╪
-blstm_0/bidirectional_rnn/fw/fw/TensorArray_1TensorArrayV3/blstm_0/bidirectional_rnn/fw/fw/strided_slice_1*
dtype0*
_output_shapes

:: *$
element_shape:         '*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*J
tensor_array_name53blstm_0/bidirectional_rnn/fw/fw/dynamic_rnn/input_0
б
8blstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeShape)blstm_0/bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:
Р
Fblstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Т
Hblstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Т
Hblstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
°
@blstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_sliceStridedSlice8blstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeFblstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackHblstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Hblstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
А
>blstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
А
>blstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
┤
8blstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/rangeRange>blstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/start@blstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice>blstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/delta*#
_output_shapes
:         *

Tidx0
Ц
Zblstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3-blstm_0/bidirectional_rnn/fw/fw/TensorArray_18blstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/range)blstm_0/bidirectional_rnn/fw/fw/transpose/blstm_0/bidirectional_rnn/fw/fw/TensorArray_1:1*
T0*<
_class2
0.loc:@blstm_0/bidirectional_rnn/fw/fw/transpose*
_output_shapes
: 
k
)blstm_0/bidirectional_rnn/fw/fw/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
г
'blstm_0/bidirectional_rnn/fw/fw/MaximumMaximum)blstm_0/bidirectional_rnn/fw/fw/Maximum/x#blstm_0/bidirectional_rnn/fw/fw/Max*
_output_shapes
: *
T0
н
'blstm_0/bidirectional_rnn/fw/fw/MinimumMinimum/blstm_0/bidirectional_rnn/fw/fw/strided_slice_1'blstm_0/bidirectional_rnn/fw/fw/Maximum*
T0*
_output_shapes
: 
y
7blstm_0/bidirectional_rnn/fw/fw/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Б
+blstm_0/bidirectional_rnn/fw/fw/while/EnterEnter7blstm_0/bidirectional_rnn/fw/fw/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
Ё
-blstm_0/bidirectional_rnn/fw/fw/while/Enter_1Enter$blstm_0/bidirectional_rnn/fw/fw/time*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
∙
-blstm_0/bidirectional_rnn/fw/fw/while/Enter_2Enter-blstm_0/bidirectional_rnn/fw/fw/TensorArray:1*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
н
-blstm_0/bidirectional_rnn/fw/fw/while/Enter_3EnterOblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*(
_output_shapes
:         А
п
-blstm_0/bidirectional_rnn/fw/fw/while/Enter_4EnterQblstm_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*(
_output_shapes
:         А
┬
+blstm_0/bidirectional_rnn/fw/fw/while/MergeMerge+blstm_0/bidirectional_rnn/fw/fw/while/Enter3blstm_0/bidirectional_rnn/fw/fw/while/NextIteration*
T0*
N*
_output_shapes
: : 
╚
-blstm_0/bidirectional_rnn/fw/fw/while/Merge_1Merge-blstm_0/bidirectional_rnn/fw/fw/while/Enter_15blstm_0/bidirectional_rnn/fw/fw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
╚
-blstm_0/bidirectional_rnn/fw/fw/while/Merge_2Merge-blstm_0/bidirectional_rnn/fw/fw/while/Enter_25blstm_0/bidirectional_rnn/fw/fw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
┌
-blstm_0/bidirectional_rnn/fw/fw/while/Merge_3Merge-blstm_0/bidirectional_rnn/fw/fw/while/Enter_35blstm_0/bidirectional_rnn/fw/fw/while/NextIteration_3*
T0*
N**
_output_shapes
:         А: 
┌
-blstm_0/bidirectional_rnn/fw/fw/while/Merge_4Merge-blstm_0/bidirectional_rnn/fw/fw/while/Enter_45blstm_0/bidirectional_rnn/fw/fw/while/NextIteration_4*
N**
_output_shapes
:         А: *
T0
■
0blstm_0/bidirectional_rnn/fw/fw/while/Less/EnterEnter/blstm_0/bidirectional_rnn/fw/fw/strided_slice_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
▓
*blstm_0/bidirectional_rnn/fw/fw/while/LessLess+blstm_0/bidirectional_rnn/fw/fw/while/Merge0blstm_0/bidirectional_rnn/fw/fw/while/Less/Enter*
T0*
_output_shapes
: 
°
2blstm_0/bidirectional_rnn/fw/fw/while/Less_1/EnterEnter'blstm_0/bidirectional_rnn/fw/fw/Minimum*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
╕
,blstm_0/bidirectional_rnn/fw/fw/while/Less_1Less-blstm_0/bidirectional_rnn/fw/fw/while/Merge_12blstm_0/bidirectional_rnn/fw/fw/while/Less_1/Enter*
T0*
_output_shapes
: 
░
0blstm_0/bidirectional_rnn/fw/fw/while/LogicalAnd
LogicalAnd*blstm_0/bidirectional_rnn/fw/fw/while/Less,blstm_0/bidirectional_rnn/fw/fw/while/Less_1*
_output_shapes
: 
Д
.blstm_0/bidirectional_rnn/fw/fw/while/LoopCondLoopCond0blstm_0/bidirectional_rnn/fw/fw/while/LogicalAnd*
_output_shapes
: 
Ў
,blstm_0/bidirectional_rnn/fw/fw/while/SwitchSwitch+blstm_0/bidirectional_rnn/fw/fw/while/Merge.blstm_0/bidirectional_rnn/fw/fw/while/LoopCond*
_output_shapes
: : *
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/fw/while/Merge
№
.blstm_0/bidirectional_rnn/fw/fw/while/Switch_1Switch-blstm_0/bidirectional_rnn/fw/fw/while/Merge_1.blstm_0/bidirectional_rnn/fw/fw/while/LoopCond*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/fw/while/Merge_1*
_output_shapes
: : 
№
.blstm_0/bidirectional_rnn/fw/fw/while/Switch_2Switch-blstm_0/bidirectional_rnn/fw/fw/while/Merge_2.blstm_0/bidirectional_rnn/fw/fw/while/LoopCond*
_output_shapes
: : *
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/fw/while/Merge_2
а
.blstm_0/bidirectional_rnn/fw/fw/while/Switch_3Switch-blstm_0/bidirectional_rnn/fw/fw/while/Merge_3.blstm_0/bidirectional_rnn/fw/fw/while/LoopCond*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/fw/while/Merge_3*<
_output_shapes*
(:         А:         А
а
.blstm_0/bidirectional_rnn/fw/fw/while/Switch_4Switch-blstm_0/bidirectional_rnn/fw/fw/while/Merge_4.blstm_0/bidirectional_rnn/fw/fw/while/LoopCond*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/fw/while/Merge_4*<
_output_shapes*
(:         А:         А
Л
.blstm_0/bidirectional_rnn/fw/fw/while/IdentityIdentity.blstm_0/bidirectional_rnn/fw/fw/while/Switch:1*
_output_shapes
: *
T0
П
0blstm_0/bidirectional_rnn/fw/fw/while/Identity_1Identity0blstm_0/bidirectional_rnn/fw/fw/while/Switch_1:1*
T0*
_output_shapes
: 
П
0blstm_0/bidirectional_rnn/fw/fw/while/Identity_2Identity0blstm_0/bidirectional_rnn/fw/fw/while/Switch_2:1*
T0*
_output_shapes
: 
б
0blstm_0/bidirectional_rnn/fw/fw/while/Identity_3Identity0blstm_0/bidirectional_rnn/fw/fw/while/Switch_3:1*(
_output_shapes
:         А*
T0
б
0blstm_0/bidirectional_rnn/fw/fw/while/Identity_4Identity0blstm_0/bidirectional_rnn/fw/fw/while/Switch_4:1*
T0*(
_output_shapes
:         А
Ю
+blstm_0/bidirectional_rnn/fw/fw/while/add/yConst/^blstm_0/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
о
)blstm_0/bidirectional_rnn/fw/fw/while/addAdd.blstm_0/bidirectional_rnn/fw/fw/while/Identity+blstm_0/bidirectional_rnn/fw/fw/while/add/y*
_output_shapes
: *
T0
Н
=blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterEnter-blstm_0/bidirectional_rnn/fw/fw/TensorArray_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╕
?blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1EnterZblstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
┤
7blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3TensorArrayReadV3=blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter0blstm_0/bidirectional_rnn/fw/fw/while/Identity_1?blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:         '
Д
8blstm_0/bidirectional_rnn/fw/fw/while/GreaterEqual/EnterEnter+blstm_0/bidirectional_rnn/fw/fw/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╤
2blstm_0/bidirectional_rnn/fw/fw/while/GreaterEqualGreaterEqual0blstm_0/bidirectional_rnn/fw/fw/while/Identity_18blstm_0/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter*
_output_shapes
:*
T0
н
7blstm_0/bidirectional_rnn/fw/fw/while/dropout/keep_probConst/^blstm_0/bidirectional_rnn/fw/fw/while/Identity*
valueB
 *fff?*
dtype0*
_output_shapes
: 
к
3blstm_0/bidirectional_rnn/fw/fw/while/dropout/ShapeShape7blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
╢
@blstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/minConst/^blstm_0/bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *    
╢
@blstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/maxConst/^blstm_0/bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
щ
Jblstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniformRandomUniform3blstm_0/bidirectional_rnn/fw/fw/while/dropout/Shape*
T0*
dtype0*
seed2Ж*'
_output_shapes
:         '*

seed 
ь
@blstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/subSub@blstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/max@blstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/min*
_output_shapes
: *
T0
З
@blstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/mulMulJblstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniform@blstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/sub*'
_output_shapes
:         '*
T0
∙
<blstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniformAdd@blstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/mul@blstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/min*
T0*'
_output_shapes
:         '
с
1blstm_0/bidirectional_rnn/fw/fw/while/dropout/addAdd7blstm_0/bidirectional_rnn/fw/fw/while/dropout/keep_prob<blstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform*
T0*'
_output_shapes
:         '
б
3blstm_0/bidirectional_rnn/fw/fw/while/dropout/FloorFloor1blstm_0/bidirectional_rnn/fw/fw/while/dropout/add*
T0*'
_output_shapes
:         '
р
1blstm_0/bidirectional_rnn/fw/fw/while/dropout/divRealDiv7blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV37blstm_0/bidirectional_rnn/fw/fw/while/dropout/keep_prob*
T0*'
_output_shapes
:         '
╥
1blstm_0/bidirectional_rnn/fw/fw/while/dropout/mulMul1blstm_0/bidirectional_rnn/fw/fw/while/dropout/div3blstm_0/bidirectional_rnn/fw/fw/while/dropout/Floor*
T0*'
_output_shapes
:         '
с
Nblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
valueB"з      
╙
Lblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
valueB
 *ЙД└╜
╙
Lblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
valueB
 *ЙД└=*
dtype0*
_output_shapes
: 
╦
Vblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformNblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/shape*
seed2С*
dtype0* 
_output_shapes
:
зА*

seed *
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel
╥
Lblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/subSubLblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/maxLblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
_output_shapes
: 
ц
Lblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/mulMulVblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformLblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
зА*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel
╪
Hblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniformAddLblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/mulLblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel* 
_output_shapes
:
зА
ч
-blstm_0/bidirectional_rnn/fw/lstm_cell/kernel
VariableV2*
dtype0* 
_output_shapes
:
зА*
shared_name *@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
	container *
shape:
зА
═
4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/AssignAssign-blstm_0/bidirectional_rnn/fw/lstm_cell/kernelHblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА
Ш
2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/readIdentity-blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
T0* 
_output_shapes
:
зА
╠
=blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zerosConst*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
┘
+blstm_0/bidirectional_rnn/fw/lstm_cell/bias
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias
╖
2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/AssignAssign+blstm_0/bidirectional_rnn/fw/lstm_cell/bias=blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias
П
0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/readIdentity+blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
_output_shapes	
:А*
T0
о
;blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat/axisConst/^blstm_0/bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
м
6blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concatConcatV21blstm_0/bidirectional_rnn/fw/fw/while/dropout/mul0blstm_0/bidirectional_rnn/fw/fw/while/Identity_4;blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat/axis*

Tidx0*
T0*
N*(
_output_shapes
:         з
Ч
<blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/EnterEnter2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context* 
_output_shapes
:
зА
П
6blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMulMatMul6blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat<blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter*
transpose_a( *(
_output_shapes
:         А*
transpose_b( *
T0
С
=blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/EnterEnter0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/read*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes	
:А*
T0*
is_constant(
Г
7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAddBiasAdd6blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul=blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:         А
и
5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/ConstConst/^blstm_0/bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
▓
?blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split/split_dimConst/^blstm_0/bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
╕
5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/splitSplit?blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split/split_dim7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd*
T0*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А
л
5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add/yConst/^blstm_0/bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
▌
3blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/addAdd7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split:25blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add/y*(
_output_shapes
:         А*
T0
к
7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/SigmoidSigmoid3blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add*
T0*(
_output_shapes
:         А
╪
3blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mulMul7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid0blstm_0/bidirectional_rnn/fw/fw/while/Identity_3*
T0*(
_output_shapes
:         А
о
9blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1Sigmoid5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split*(
_output_shapes
:         А*
T0
и
4blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/TanhTanh7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split:1*
T0*(
_output_shapes
:         А
р
5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1Mul9blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_14blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh*(
_output_shapes
:         А*
T0
█
5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1Add3blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1*
T0*(
_output_shapes
:         А
░
9blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2Sigmoid7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split:3*(
_output_shapes
:         А*
T0
и
6blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1Tanh5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1*(
_output_shapes
:         А*
T0
т
5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2Mul9blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_26blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*
T0*(
_output_shapes
:         А
╥
2blstm_0/bidirectional_rnn/fw/fw/while/Select/EnterEnter%blstm_0/bidirectional_rnn/fw/fw/zeros*
is_constant(*(
_output_shapes
:         А*C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
parallel_iterations 
╥
,blstm_0/bidirectional_rnn/fw/fw/while/SelectSelect2blstm_0/bidirectional_rnn/fw/fw/while/GreaterEqual2blstm_0/bidirectional_rnn/fw/fw/while/Select/Enter5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*(
_output_shapes
:         А
╥
.blstm_0/bidirectional_rnn/fw/fw/while/Select_1Select2blstm_0/bidirectional_rnn/fw/fw/while/GreaterEqual0blstm_0/bidirectional_rnn/fw/fw/while/Identity_35blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1*(
_output_shapes
:         А
╥
.blstm_0/bidirectional_rnn/fw/fw/while/Select_2Select2blstm_0/bidirectional_rnn/fw/fw/while/GreaterEqual0blstm_0/bidirectional_rnn/fw/fw/while/Identity_45blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*(
_output_shapes
:         А
ч
Oblstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter+blstm_0/bidirectional_rnn/fw/fw/TensorArray*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context
н
Iblstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Oblstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter0blstm_0/bidirectional_rnn/fw/fw/while/Identity_1,blstm_0/bidirectional_rnn/fw/fw/while/Select0blstm_0/bidirectional_rnn/fw/fw/while/Identity_2*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
_output_shapes
: 
а
-blstm_0/bidirectional_rnn/fw/fw/while/add_1/yConst/^blstm_0/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
┤
+blstm_0/bidirectional_rnn/fw/fw/while/add_1Add0blstm_0/bidirectional_rnn/fw/fw/while/Identity_1-blstm_0/bidirectional_rnn/fw/fw/while/add_1/y*
T0*
_output_shapes
: 
Р
3blstm_0/bidirectional_rnn/fw/fw/while/NextIterationNextIteration)blstm_0/bidirectional_rnn/fw/fw/while/add*
T0*
_output_shapes
: 
Ф
5blstm_0/bidirectional_rnn/fw/fw/while/NextIteration_1NextIteration+blstm_0/bidirectional_rnn/fw/fw/while/add_1*
_output_shapes
: *
T0
▓
5blstm_0/bidirectional_rnn/fw/fw/while/NextIteration_2NextIterationIblstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
й
5blstm_0/bidirectional_rnn/fw/fw/while/NextIteration_3NextIteration.blstm_0/bidirectional_rnn/fw/fw/while/Select_1*
T0*(
_output_shapes
:         А
й
5blstm_0/bidirectional_rnn/fw/fw/while/NextIteration_4NextIteration.blstm_0/bidirectional_rnn/fw/fw/while/Select_2*
T0*(
_output_shapes
:         А
Б
*blstm_0/bidirectional_rnn/fw/fw/while/ExitExit,blstm_0/bidirectional_rnn/fw/fw/while/Switch*
T0*
_output_shapes
: 
Е
,blstm_0/bidirectional_rnn/fw/fw/while/Exit_1Exit.blstm_0/bidirectional_rnn/fw/fw/while/Switch_1*
T0*
_output_shapes
: 
Е
,blstm_0/bidirectional_rnn/fw/fw/while/Exit_2Exit.blstm_0/bidirectional_rnn/fw/fw/while/Switch_2*
T0*
_output_shapes
: 
Ч
,blstm_0/bidirectional_rnn/fw/fw/while/Exit_3Exit.blstm_0/bidirectional_rnn/fw/fw/while/Switch_3*
T0*(
_output_shapes
:         А
Ч
,blstm_0/bidirectional_rnn/fw/fw/while/Exit_4Exit.blstm_0/bidirectional_rnn/fw/fw/while/Switch_4*(
_output_shapes
:         А*
T0
К
Bblstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3+blstm_0/bidirectional_rnn/fw/fw/TensorArray,blstm_0/bidirectional_rnn/fw/fw/while/Exit_2*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/fw/TensorArray*
_output_shapes
: 
╛
<blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/range/startConst*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/fw/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
╛
<blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/range/deltaConst*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/fw/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
Ё
6blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/rangeRange<blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/range/startBblstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3<blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/range/delta*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/fw/TensorArray*#
_output_shapes
:         *

Tidx0
Щ
Dblstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3+blstm_0/bidirectional_rnn/fw/fw/TensorArray6blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/range,blstm_0/bidirectional_rnn/fw/fw/while/Exit_2*
dtype0*5
_output_shapes#
!:                  А*%
element_shape:         А*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/fw/TensorArray
r
'blstm_0/bidirectional_rnn/fw/fw/Const_2Const*
valueB:А*
dtype0*
_output_shapes
:
h
&blstm_0/bidirectional_rnn/fw/fw/Rank_3Const*
value	B :*
dtype0*
_output_shapes
: 
o
-blstm_0/bidirectional_rnn/fw/fw/range_3/startConst*
value	B :*
dtype0*
_output_shapes
: 
o
-blstm_0/bidirectional_rnn/fw/fw/range_3/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
▐
'blstm_0/bidirectional_rnn/fw/fw/range_3Range-blstm_0/bidirectional_rnn/fw/fw/range_3/start&blstm_0/bidirectional_rnn/fw/fw/Rank_3-blstm_0/bidirectional_rnn/fw/fw/range_3/delta*
_output_shapes
:*

Tidx0
В
1blstm_0/bidirectional_rnn/fw/fw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
o
-blstm_0/bidirectional_rnn/fw/fw/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
∙
(blstm_0/bidirectional_rnn/fw/fw/concat_2ConcatV21blstm_0/bidirectional_rnn/fw/fw/concat_2/values_0'blstm_0/bidirectional_rnn/fw/fw/range_3-blstm_0/bidirectional_rnn/fw/fw/concat_2/axis*
N*
_output_shapes
:*

Tidx0*
T0
ї
+blstm_0/bidirectional_rnn/fw/fw/transpose_1	TransposeDblstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3(blstm_0/bidirectional_rnn/fw/fw/concat_2*
Tperm0*
T0*5
_output_shapes#
!:                  А
╘
,blstm_0/bidirectional_rnn/bw/ReverseSequenceReverseSequenceinputs/Placeholderinputs/Placeholder_2*
seq_dim*

Tlen0*4
_output_shapes"
 :                  '*
	batch_dim *
T0
f
$blstm_0/bidirectional_rnn/bw/bw/RankConst*
value	B :*
dtype0*
_output_shapes
: 
m
+blstm_0/bidirectional_rnn/bw/bw/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
m
+blstm_0/bidirectional_rnn/bw/bw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
╓
%blstm_0/bidirectional_rnn/bw/bw/rangeRange+blstm_0/bidirectional_rnn/bw/bw/range/start$blstm_0/bidirectional_rnn/bw/bw/Rank+blstm_0/bidirectional_rnn/bw/bw/range/delta*

Tidx0*
_output_shapes
:
А
/blstm_0/bidirectional_rnn/bw/bw/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
m
+blstm_0/bidirectional_rnn/bw/bw/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ё
&blstm_0/bidirectional_rnn/bw/bw/concatConcatV2/blstm_0/bidirectional_rnn/bw/bw/concat/values_0%blstm_0/bidirectional_rnn/bw/bw/range+blstm_0/bidirectional_rnn/bw/bw/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
╪
)blstm_0/bidirectional_rnn/bw/bw/transpose	Transpose,blstm_0/bidirectional_rnn/bw/ReverseSequence&blstm_0/bidirectional_rnn/bw/bw/concat*
Tperm0*
T0*4
_output_shapes"
 :                  '
t
/blstm_0/bidirectional_rnn/bw/bw/sequence_lengthIdentityinputs/Placeholder_2*
T0*
_output_shapes
:
О
%blstm_0/bidirectional_rnn/bw/bw/ShapeShape)blstm_0/bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:
}
3blstm_0/bidirectional_rnn/bw/bw/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:

5blstm_0/bidirectional_rnn/bw/bw/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

5blstm_0/bidirectional_rnn/bw/bw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Щ
-blstm_0/bidirectional_rnn/bw/bw/strided_sliceStridedSlice%blstm_0/bidirectional_rnn/bw/bw/Shape3blstm_0/bidirectional_rnn/bw/bw/strided_slice/stack5blstm_0/bidirectional_rnn/bw/bw/strided_slice/stack_15blstm_0/bidirectional_rnn/bw/bw/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
Ъ
Xblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Ь
Tblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims
ExpandDims-blstm_0/bidirectional_rnn/bw/bw/strided_sliceXblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
Ъ
Oblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ConstConst*
valueB:А*
dtype0*
_output_shapes
:
Ч
Ublstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ф
Pblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concatConcatV2Tblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDimsOblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ConstUblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Ъ
Ublstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
┼
Oblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zerosFillPblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concatUblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros/Const*(
_output_shapes
:         А*
T0*

index_type0
Ь
Zblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
а
Vblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1
ExpandDims-blstm_0/bidirectional_rnn/bw/bw/strided_sliceZblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dim*
_output_shapes
:*

Tdim0*
T0
Ь
Qblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Const*
valueB:А*
dtype0*
_output_shapes
:
Ь
Zblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dimConst*
dtype0*
_output_shapes
: *
value	B : 
а
Vblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2
ExpandDims-blstm_0/bidirectional_rnn/bw/bw/strided_sliceZblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0*
_output_shapes
:
Ь
Qblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Const*
valueB:А*
dtype0*
_output_shapes
:
Щ
Wblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ь
Rblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2Vblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2Qblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Wblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ь
Wblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╦
Qblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1FillRblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1Wblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/Const*(
_output_shapes
:         А*
T0*

index_type0
Ь
Zblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
а
Vblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3
ExpandDims-blstm_0/bidirectional_rnn/bw/bw/strided_sliceZblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dim*
T0*
_output_shapes
:*

Tdim0
Ь
Qblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/Const_3Const*
valueB:А*
dtype0*
_output_shapes
:
Я
'blstm_0/bidirectional_rnn/bw/bw/Shape_1Shape/blstm_0/bidirectional_rnn/bw/bw/sequence_length*
T0*
out_type0*#
_output_shapes
:         
Ц
%blstm_0/bidirectional_rnn/bw/bw/stackPack-blstm_0/bidirectional_rnn/bw/bw/strided_slice*
N*
_output_shapes
:*
T0*

axis 
м
%blstm_0/bidirectional_rnn/bw/bw/EqualEqual'blstm_0/bidirectional_rnn/bw/bw/Shape_1%blstm_0/bidirectional_rnn/bw/bw/stack*
T0*#
_output_shapes
:         
o
%blstm_0/bidirectional_rnn/bw/bw/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
н
#blstm_0/bidirectional_rnn/bw/bw/AllAll%blstm_0/bidirectional_rnn/bw/bw/Equal%blstm_0/bidirectional_rnn/bw/bw/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
╝
,blstm_0/bidirectional_rnn/bw/bw/Assert/ConstConst*`
valueWBU BOExpected shape for Tensor blstm_0/bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 

.blstm_0/bidirectional_rnn/bw/bw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
─
4blstm_0/bidirectional_rnn/bw/bw/Assert/Assert/data_0Const*
dtype0*
_output_shapes
: *`
valueWBU BOExpected shape for Tensor blstm_0/bidirectional_rnn/bw/bw/sequence_length:0 is 
Е
4blstm_0/bidirectional_rnn/bw/bw/Assert/Assert/data_2Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
╕
-blstm_0/bidirectional_rnn/bw/bw/Assert/AssertAssert#blstm_0/bidirectional_rnn/bw/bw/All4blstm_0/bidirectional_rnn/bw/bw/Assert/Assert/data_0%blstm_0/bidirectional_rnn/bw/bw/stack4blstm_0/bidirectional_rnn/bw/bw/Assert/Assert/data_2'blstm_0/bidirectional_rnn/bw/bw/Shape_1*
T
2*
	summarize
╗
+blstm_0/bidirectional_rnn/bw/bw/CheckSeqLenIdentity/blstm_0/bidirectional_rnn/bw/bw/sequence_length.^blstm_0/bidirectional_rnn/bw/bw/Assert/Assert*
T0*
_output_shapes
:
Р
'blstm_0/bidirectional_rnn/bw/bw/Shape_2Shape)blstm_0/bidirectional_rnn/bw/bw/transpose*
_output_shapes
:*
T0*
out_type0

5blstm_0/bidirectional_rnn/bw/bw/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
Б
7blstm_0/bidirectional_rnn/bw/bw/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Б
7blstm_0/bidirectional_rnn/bw/bw/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
г
/blstm_0/bidirectional_rnn/bw/bw/strided_slice_1StridedSlice'blstm_0/bidirectional_rnn/bw/bw/Shape_25blstm_0/bidirectional_rnn/bw/bw/strided_slice_1/stack7blstm_0/bidirectional_rnn/bw/bw/strided_slice_1/stack_17blstm_0/bidirectional_rnn/bw/bw/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Р
'blstm_0/bidirectional_rnn/bw/bw/Shape_3Shape)blstm_0/bidirectional_rnn/bw/bw/transpose*
_output_shapes
:*
T0*
out_type0

5blstm_0/bidirectional_rnn/bw/bw/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
Б
7blstm_0/bidirectional_rnn/bw/bw/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Б
7blstm_0/bidirectional_rnn/bw/bw/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
г
/blstm_0/bidirectional_rnn/bw/bw/strided_slice_2StridedSlice'blstm_0/bidirectional_rnn/bw/bw/Shape_35blstm_0/bidirectional_rnn/bw/bw/strided_slice_2/stack7blstm_0/bidirectional_rnn/bw/bw/strided_slice_2/stack_17blstm_0/bidirectional_rnn/bw/bw/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
p
.blstm_0/bidirectional_rnn/bw/bw/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
╩
*blstm_0/bidirectional_rnn/bw/bw/ExpandDims
ExpandDims/blstm_0/bidirectional_rnn/bw/bw/strided_slice_2.blstm_0/bidirectional_rnn/bw/bw/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
r
'blstm_0/bidirectional_rnn/bw/bw/Const_1Const*
valueB:А*
dtype0*
_output_shapes
:
o
-blstm_0/bidirectional_rnn/bw/bw/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Є
(blstm_0/bidirectional_rnn/bw/bw/concat_1ConcatV2*blstm_0/bidirectional_rnn/bw/bw/ExpandDims'blstm_0/bidirectional_rnn/bw/bw/Const_1-blstm_0/bidirectional_rnn/bw/bw/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
p
+blstm_0/bidirectional_rnn/bw/bw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╔
%blstm_0/bidirectional_rnn/bw/bw/zerosFill(blstm_0/bidirectional_rnn/bw/bw/concat_1+blstm_0/bidirectional_rnn/bw/bw/zeros/Const*
T0*

index_type0*(
_output_shapes
:         А
|
&blstm_0/bidirectional_rnn/bw/bw/Rank_1Rank+blstm_0/bidirectional_rnn/bw/bw/CheckSeqLen*
T0*
_output_shapes
: 
o
-blstm_0/bidirectional_rnn/bw/bw/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
o
-blstm_0/bidirectional_rnn/bw/bw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ч
'blstm_0/bidirectional_rnn/bw/bw/range_1Range-blstm_0/bidirectional_rnn/bw/bw/range_1/start&blstm_0/bidirectional_rnn/bw/bw/Rank_1-blstm_0/bidirectional_rnn/bw/bw/range_1/delta*

Tidx0*#
_output_shapes
:         
╛
#blstm_0/bidirectional_rnn/bw/bw/MinMin+blstm_0/bidirectional_rnn/bw/bw/CheckSeqLen'blstm_0/bidirectional_rnn/bw/bw/range_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
|
&blstm_0/bidirectional_rnn/bw/bw/Rank_2Rank+blstm_0/bidirectional_rnn/bw/bw/CheckSeqLen*
T0*
_output_shapes
: 
o
-blstm_0/bidirectional_rnn/bw/bw/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
o
-blstm_0/bidirectional_rnn/bw/bw/range_2/deltaConst*
dtype0*
_output_shapes
: *
value	B :
ч
'blstm_0/bidirectional_rnn/bw/bw/range_2Range-blstm_0/bidirectional_rnn/bw/bw/range_2/start&blstm_0/bidirectional_rnn/bw/bw/Rank_2-blstm_0/bidirectional_rnn/bw/bw/range_2/delta*#
_output_shapes
:         *

Tidx0
╛
#blstm_0/bidirectional_rnn/bw/bw/MaxMax+blstm_0/bidirectional_rnn/bw/bw/CheckSeqLen'blstm_0/bidirectional_rnn/bw/bw/range_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
f
$blstm_0/bidirectional_rnn/bw/bw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
╪
+blstm_0/bidirectional_rnn/bw/bw/TensorArrayTensorArrayV3/blstm_0/bidirectional_rnn/bw/bw/strided_slice_1*K
tensor_array_name64blstm_0/bidirectional_rnn/bw/bw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *%
element_shape:         А*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
╪
-blstm_0/bidirectional_rnn/bw/bw/TensorArray_1TensorArrayV3/blstm_0/bidirectional_rnn/bw/bw/strided_slice_1*
dtype0*
_output_shapes

:: *$
element_shape:         '*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*J
tensor_array_name53blstm_0/bidirectional_rnn/bw/bw/dynamic_rnn/input_0
б
8blstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeShape)blstm_0/bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:
Р
Fblstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Т
Hblstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Т
Hblstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
°
@blstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_sliceStridedSlice8blstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeFblstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackHblstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Hblstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
А
>blstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
А
>blstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
┤
8blstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/rangeRange>blstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/start@blstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice>blstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/delta*#
_output_shapes
:         *

Tidx0
Ц
Zblstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3-blstm_0/bidirectional_rnn/bw/bw/TensorArray_18blstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/range)blstm_0/bidirectional_rnn/bw/bw/transpose/blstm_0/bidirectional_rnn/bw/bw/TensorArray_1:1*
T0*<
_class2
0.loc:@blstm_0/bidirectional_rnn/bw/bw/transpose*
_output_shapes
: 
k
)blstm_0/bidirectional_rnn/bw/bw/Maximum/xConst*
dtype0*
_output_shapes
: *
value	B :
г
'blstm_0/bidirectional_rnn/bw/bw/MaximumMaximum)blstm_0/bidirectional_rnn/bw/bw/Maximum/x#blstm_0/bidirectional_rnn/bw/bw/Max*
T0*
_output_shapes
: 
н
'blstm_0/bidirectional_rnn/bw/bw/MinimumMinimum/blstm_0/bidirectional_rnn/bw/bw/strided_slice_1'blstm_0/bidirectional_rnn/bw/bw/Maximum*
T0*
_output_shapes
: 
y
7blstm_0/bidirectional_rnn/bw/bw/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Б
+blstm_0/bidirectional_rnn/bw/bw/while/EnterEnter7blstm_0/bidirectional_rnn/bw/bw/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
Ё
-blstm_0/bidirectional_rnn/bw/bw/while/Enter_1Enter$blstm_0/bidirectional_rnn/bw/bw/time*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0*
is_constant( 
∙
-blstm_0/bidirectional_rnn/bw/bw/while/Enter_2Enter-blstm_0/bidirectional_rnn/bw/bw/TensorArray:1*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0*
is_constant( 
н
-blstm_0/bidirectional_rnn/bw/bw/while/Enter_3EnterOblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*(
_output_shapes
:         А
п
-blstm_0/bidirectional_rnn/bw/bw/while/Enter_4EnterQblstm_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*(
_output_shapes
:         А
┬
+blstm_0/bidirectional_rnn/bw/bw/while/MergeMerge+blstm_0/bidirectional_rnn/bw/bw/while/Enter3blstm_0/bidirectional_rnn/bw/bw/while/NextIteration*
T0*
N*
_output_shapes
: : 
╚
-blstm_0/bidirectional_rnn/bw/bw/while/Merge_1Merge-blstm_0/bidirectional_rnn/bw/bw/while/Enter_15blstm_0/bidirectional_rnn/bw/bw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
╚
-blstm_0/bidirectional_rnn/bw/bw/while/Merge_2Merge-blstm_0/bidirectional_rnn/bw/bw/while/Enter_25blstm_0/bidirectional_rnn/bw/bw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
┌
-blstm_0/bidirectional_rnn/bw/bw/while/Merge_3Merge-blstm_0/bidirectional_rnn/bw/bw/while/Enter_35blstm_0/bidirectional_rnn/bw/bw/while/NextIteration_3*
N**
_output_shapes
:         А: *
T0
┌
-blstm_0/bidirectional_rnn/bw/bw/while/Merge_4Merge-blstm_0/bidirectional_rnn/bw/bw/while/Enter_45blstm_0/bidirectional_rnn/bw/bw/while/NextIteration_4*
N**
_output_shapes
:         А: *
T0
■
0blstm_0/bidirectional_rnn/bw/bw/while/Less/EnterEnter/blstm_0/bidirectional_rnn/bw/bw/strided_slice_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
▓
*blstm_0/bidirectional_rnn/bw/bw/while/LessLess+blstm_0/bidirectional_rnn/bw/bw/while/Merge0blstm_0/bidirectional_rnn/bw/bw/while/Less/Enter*
T0*
_output_shapes
: 
°
2blstm_0/bidirectional_rnn/bw/bw/while/Less_1/EnterEnter'blstm_0/bidirectional_rnn/bw/bw/Minimum*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
╕
,blstm_0/bidirectional_rnn/bw/bw/while/Less_1Less-blstm_0/bidirectional_rnn/bw/bw/while/Merge_12blstm_0/bidirectional_rnn/bw/bw/while/Less_1/Enter*
T0*
_output_shapes
: 
░
0blstm_0/bidirectional_rnn/bw/bw/while/LogicalAnd
LogicalAnd*blstm_0/bidirectional_rnn/bw/bw/while/Less,blstm_0/bidirectional_rnn/bw/bw/while/Less_1*
_output_shapes
: 
Д
.blstm_0/bidirectional_rnn/bw/bw/while/LoopCondLoopCond0blstm_0/bidirectional_rnn/bw/bw/while/LogicalAnd*
_output_shapes
: 
Ў
,blstm_0/bidirectional_rnn/bw/bw/while/SwitchSwitch+blstm_0/bidirectional_rnn/bw/bw/while/Merge.blstm_0/bidirectional_rnn/bw/bw/while/LoopCond*
_output_shapes
: : *
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/bw/while/Merge
№
.blstm_0/bidirectional_rnn/bw/bw/while/Switch_1Switch-blstm_0/bidirectional_rnn/bw/bw/while/Merge_1.blstm_0/bidirectional_rnn/bw/bw/while/LoopCond*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/bw/while/Merge_1*
_output_shapes
: : 
№
.blstm_0/bidirectional_rnn/bw/bw/while/Switch_2Switch-blstm_0/bidirectional_rnn/bw/bw/while/Merge_2.blstm_0/bidirectional_rnn/bw/bw/while/LoopCond*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/bw/while/Merge_2*
_output_shapes
: : 
а
.blstm_0/bidirectional_rnn/bw/bw/while/Switch_3Switch-blstm_0/bidirectional_rnn/bw/bw/while/Merge_3.blstm_0/bidirectional_rnn/bw/bw/while/LoopCond*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/bw/while/Merge_3*<
_output_shapes*
(:         А:         А
а
.blstm_0/bidirectional_rnn/bw/bw/while/Switch_4Switch-blstm_0/bidirectional_rnn/bw/bw/while/Merge_4.blstm_0/bidirectional_rnn/bw/bw/while/LoopCond*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/bw/while/Merge_4*<
_output_shapes*
(:         А:         А
Л
.blstm_0/bidirectional_rnn/bw/bw/while/IdentityIdentity.blstm_0/bidirectional_rnn/bw/bw/while/Switch:1*
T0*
_output_shapes
: 
П
0blstm_0/bidirectional_rnn/bw/bw/while/Identity_1Identity0blstm_0/bidirectional_rnn/bw/bw/while/Switch_1:1*
T0*
_output_shapes
: 
П
0blstm_0/bidirectional_rnn/bw/bw/while/Identity_2Identity0blstm_0/bidirectional_rnn/bw/bw/while/Switch_2:1*
T0*
_output_shapes
: 
б
0blstm_0/bidirectional_rnn/bw/bw/while/Identity_3Identity0blstm_0/bidirectional_rnn/bw/bw/while/Switch_3:1*
T0*(
_output_shapes
:         А
б
0blstm_0/bidirectional_rnn/bw/bw/while/Identity_4Identity0blstm_0/bidirectional_rnn/bw/bw/while/Switch_4:1*
T0*(
_output_shapes
:         А
Ю
+blstm_0/bidirectional_rnn/bw/bw/while/add/yConst/^blstm_0/bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
о
)blstm_0/bidirectional_rnn/bw/bw/while/addAdd.blstm_0/bidirectional_rnn/bw/bw/while/Identity+blstm_0/bidirectional_rnn/bw/bw/while/add/y*
T0*
_output_shapes
: 
Н
=blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterEnter-blstm_0/bidirectional_rnn/bw/bw/TensorArray_1*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
╕
?blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1EnterZblstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
┤
7blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3TensorArrayReadV3=blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter0blstm_0/bidirectional_rnn/bw/bw/while/Identity_1?blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:         '
Д
8blstm_0/bidirectional_rnn/bw/bw/while/GreaterEqual/EnterEnter+blstm_0/bidirectional_rnn/bw/bw/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
╤
2blstm_0/bidirectional_rnn/bw/bw/while/GreaterEqualGreaterEqual0blstm_0/bidirectional_rnn/bw/bw/while/Identity_18blstm_0/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter*
T0*
_output_shapes
:
н
7blstm_0/bidirectional_rnn/bw/bw/while/dropout/keep_probConst/^blstm_0/bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *fff?
к
3blstm_0/bidirectional_rnn/bw/bw/while/dropout/ShapeShape7blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3*
_output_shapes
:*
T0*
out_type0
╢
@blstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/minConst/^blstm_0/bidirectional_rnn/bw/bw/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
╢
@blstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/maxConst/^blstm_0/bidirectional_rnn/bw/bw/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
щ
Jblstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniformRandomUniform3blstm_0/bidirectional_rnn/bw/bw/while/dropout/Shape*
seed2╬*'
_output_shapes
:         '*

seed *
T0*
dtype0
ь
@blstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/subSub@blstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/max@blstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/min*
_output_shapes
: *
T0
З
@blstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/mulMulJblstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniform@blstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/sub*
T0*'
_output_shapes
:         '
∙
<blstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniformAdd@blstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/mul@blstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/min*
T0*'
_output_shapes
:         '
с
1blstm_0/bidirectional_rnn/bw/bw/while/dropout/addAdd7blstm_0/bidirectional_rnn/bw/bw/while/dropout/keep_prob<blstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform*
T0*'
_output_shapes
:         '
б
3blstm_0/bidirectional_rnn/bw/bw/while/dropout/FloorFloor1blstm_0/bidirectional_rnn/bw/bw/while/dropout/add*
T0*'
_output_shapes
:         '
р
1blstm_0/bidirectional_rnn/bw/bw/while/dropout/divRealDiv7blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV37blstm_0/bidirectional_rnn/bw/bw/while/dropout/keep_prob*'
_output_shapes
:         '*
T0
╥
1blstm_0/bidirectional_rnn/bw/bw/while/dropout/mulMul1blstm_0/bidirectional_rnn/bw/bw/while/dropout/div3blstm_0/bidirectional_rnn/bw/bw/while/dropout/Floor*
T0*'
_output_shapes
:         '
с
Nblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
valueB"з      *
dtype0*
_output_shapes
:
╙
Lblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
valueB
 *ЙД└╜*
dtype0
╙
Lblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
valueB
 *ЙД└=*
dtype0*
_output_shapes
: 
╦
Vblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformNblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/shape*
seed2┘*
dtype0* 
_output_shapes
:
зА*

seed *
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel
╥
Lblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/subSubLblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/maxLblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
_output_shapes
: 
ц
Lblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/mulMulVblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformLblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
зА
╪
Hblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniformAddLblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/mulLblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
зА
ч
-blstm_0/bidirectional_rnn/bw/lstm_cell/kernel
VariableV2*
	container *
shape:
зА*
dtype0* 
_output_shapes
:
зА*
shared_name *@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel
═
4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/AssignAssign-blstm_0/bidirectional_rnn/bw/lstm_cell/kernelHblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА*
use_locking(*
T0
Ш
2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/readIdentity-blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
T0* 
_output_shapes
:
зА
╠
=blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zerosConst*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
┘
+blstm_0/bidirectional_rnn/bw/lstm_cell/bias
VariableV2*
shared_name *>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
╖
2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/AssignAssign+blstm_0/bidirectional_rnn/bw/lstm_cell/bias=blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
П
0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/readIdentity+blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
T0*
_output_shapes	
:А
о
;blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat/axisConst/^blstm_0/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
м
6blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concatConcatV21blstm_0/bidirectional_rnn/bw/bw/while/dropout/mul0blstm_0/bidirectional_rnn/bw/bw/while/Identity_4;blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat/axis*

Tidx0*
T0*
N*(
_output_shapes
:         з
Ч
<blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/EnterEnter2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context* 
_output_shapes
:
зА
П
6blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMulMatMul6blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat<blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:         А
С
=blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/EnterEnter0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes	
:А
Г
7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAddBiasAdd6blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul=blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:         А
и
5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/ConstConst/^blstm_0/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
▓
?blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split/split_dimConst/^blstm_0/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╕
5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/splitSplit?blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split/split_dim7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А*
T0
л
5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add/yConst/^blstm_0/bidirectional_rnn/bw/bw/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
▌
3blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/addAdd7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split:25blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add/y*
T0*(
_output_shapes
:         А
к
7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/SigmoidSigmoid3blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add*
T0*(
_output_shapes
:         А
╪
3blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mulMul7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid0blstm_0/bidirectional_rnn/bw/bw/while/Identity_3*
T0*(
_output_shapes
:         А
о
9blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1Sigmoid5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split*(
_output_shapes
:         А*
T0
и
4blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/TanhTanh7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split:1*(
_output_shapes
:         А*
T0
р
5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1Mul9blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_14blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*
T0*(
_output_shapes
:         А
█
5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1Add3blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1*
T0*(
_output_shapes
:         А
░
9blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2Sigmoid7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split:3*
T0*(
_output_shapes
:         А
и
6blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1Tanh5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1*
T0*(
_output_shapes
:         А
т
5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2Mul9blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_26blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*
T0*(
_output_shapes
:         А
╥
2blstm_0/bidirectional_rnn/bw/bw/while/Select/EnterEnter%blstm_0/bidirectional_rnn/bw/bw/zeros*(
_output_shapes
:         А*C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(
╥
,blstm_0/bidirectional_rnn/bw/bw/while/SelectSelect2blstm_0/bidirectional_rnn/bw/bw/while/GreaterEqual2blstm_0/bidirectional_rnn/bw/bw/while/Select/Enter5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*(
_output_shapes
:         А
╥
.blstm_0/bidirectional_rnn/bw/bw/while/Select_1Select2blstm_0/bidirectional_rnn/bw/bw/while/GreaterEqual0blstm_0/bidirectional_rnn/bw/bw/while/Identity_35blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1*(
_output_shapes
:         А
╥
.blstm_0/bidirectional_rnn/bw/bw/while/Select_2Select2blstm_0/bidirectional_rnn/bw/bw/while/GreaterEqual0blstm_0/bidirectional_rnn/bw/bw/while/Identity_45blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*(
_output_shapes
:         А*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2
ч
Oblstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter+blstm_0/bidirectional_rnn/bw/bw/TensorArray*
is_constant(*C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
parallel_iterations 
н
Iblstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Oblstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter0blstm_0/bidirectional_rnn/bw/bw/while/Identity_1,blstm_0/bidirectional_rnn/bw/bw/while/Select0blstm_0/bidirectional_rnn/bw/bw/while/Identity_2*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
_output_shapes
: 
а
-blstm_0/bidirectional_rnn/bw/bw/while/add_1/yConst/^blstm_0/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
┤
+blstm_0/bidirectional_rnn/bw/bw/while/add_1Add0blstm_0/bidirectional_rnn/bw/bw/while/Identity_1-blstm_0/bidirectional_rnn/bw/bw/while/add_1/y*
T0*
_output_shapes
: 
Р
3blstm_0/bidirectional_rnn/bw/bw/while/NextIterationNextIteration)blstm_0/bidirectional_rnn/bw/bw/while/add*
T0*
_output_shapes
: 
Ф
5blstm_0/bidirectional_rnn/bw/bw/while/NextIteration_1NextIteration+blstm_0/bidirectional_rnn/bw/bw/while/add_1*
T0*
_output_shapes
: 
▓
5blstm_0/bidirectional_rnn/bw/bw/while/NextIteration_2NextIterationIblstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
й
5blstm_0/bidirectional_rnn/bw/bw/while/NextIteration_3NextIteration.blstm_0/bidirectional_rnn/bw/bw/while/Select_1*
T0*(
_output_shapes
:         А
й
5blstm_0/bidirectional_rnn/bw/bw/while/NextIteration_4NextIteration.blstm_0/bidirectional_rnn/bw/bw/while/Select_2*
T0*(
_output_shapes
:         А
Б
*blstm_0/bidirectional_rnn/bw/bw/while/ExitExit,blstm_0/bidirectional_rnn/bw/bw/while/Switch*
T0*
_output_shapes
: 
Е
,blstm_0/bidirectional_rnn/bw/bw/while/Exit_1Exit.blstm_0/bidirectional_rnn/bw/bw/while/Switch_1*
T0*
_output_shapes
: 
Е
,blstm_0/bidirectional_rnn/bw/bw/while/Exit_2Exit.blstm_0/bidirectional_rnn/bw/bw/while/Switch_2*
T0*
_output_shapes
: 
Ч
,blstm_0/bidirectional_rnn/bw/bw/while/Exit_3Exit.blstm_0/bidirectional_rnn/bw/bw/while/Switch_3*
T0*(
_output_shapes
:         А
Ч
,blstm_0/bidirectional_rnn/bw/bw/while/Exit_4Exit.blstm_0/bidirectional_rnn/bw/bw/while/Switch_4*
T0*(
_output_shapes
:         А
К
Bblstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3+blstm_0/bidirectional_rnn/bw/bw/TensorArray,blstm_0/bidirectional_rnn/bw/bw/while/Exit_2*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
: 
╛
<blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/range/startConst*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/bw/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
╛
<blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/range/deltaConst*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/bw/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
Ё
6blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/rangeRange<blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/range/startBblstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3<blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/range/delta*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/bw/TensorArray*#
_output_shapes
:         *

Tidx0
Щ
Dblstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3+blstm_0/bidirectional_rnn/bw/bw/TensorArray6blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/range,blstm_0/bidirectional_rnn/bw/bw/while/Exit_2*%
element_shape:         А*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/bw/TensorArray*
dtype0*5
_output_shapes#
!:                  А
r
'blstm_0/bidirectional_rnn/bw/bw/Const_2Const*
valueB:А*
dtype0*
_output_shapes
:
h
&blstm_0/bidirectional_rnn/bw/bw/Rank_3Const*
value	B :*
dtype0*
_output_shapes
: 
o
-blstm_0/bidirectional_rnn/bw/bw/range_3/startConst*
value	B :*
dtype0*
_output_shapes
: 
o
-blstm_0/bidirectional_rnn/bw/bw/range_3/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
▐
'blstm_0/bidirectional_rnn/bw/bw/range_3Range-blstm_0/bidirectional_rnn/bw/bw/range_3/start&blstm_0/bidirectional_rnn/bw/bw/Rank_3-blstm_0/bidirectional_rnn/bw/bw/range_3/delta*
_output_shapes
:*

Tidx0
В
1blstm_0/bidirectional_rnn/bw/bw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
o
-blstm_0/bidirectional_rnn/bw/bw/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
∙
(blstm_0/bidirectional_rnn/bw/bw/concat_2ConcatV21blstm_0/bidirectional_rnn/bw/bw/concat_2/values_0'blstm_0/bidirectional_rnn/bw/bw/range_3-blstm_0/bidirectional_rnn/bw/bw/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
ї
+blstm_0/bidirectional_rnn/bw/bw/transpose_1	TransposeDblstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3(blstm_0/bidirectional_rnn/bw/bw/concat_2*
T0*5
_output_shapes#
!:                  А*
Tperm0
┘
blstm_0/ReverseSequenceReverseSequence+blstm_0/bidirectional_rnn/bw/bw/transpose_1inputs/Placeholder_2*
	batch_dim *
T0*
seq_dim*

Tlen0*5
_output_shapes#
!:                  А
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
║
concatConcatV2+blstm_0/bidirectional_rnn/fw/fw/transpose_1blstm_0/ReverseSequenceconcat/axis*
T0*
N*5
_output_shapes#
!:                  А*

Tidx0
e
 blstm_1/DropoutWrapperInit/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
g
"blstm_1/DropoutWrapperInit/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"blstm_1/DropoutWrapperInit/Const_2Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"blstm_1/DropoutWrapperInit_1/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
i
$blstm_1/DropoutWrapperInit_1/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
i
$blstm_1/DropoutWrapperInit_1/Const_2Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
$blstm_1/bidirectional_rnn/fw/fw/RankConst*
value	B :*
dtype0*
_output_shapes
: 
m
+blstm_1/bidirectional_rnn/fw/fw/range/startConst*
dtype0*
_output_shapes
: *
value	B :
m
+blstm_1/bidirectional_rnn/fw/fw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
╓
%blstm_1/bidirectional_rnn/fw/fw/rangeRange+blstm_1/bidirectional_rnn/fw/fw/range/start$blstm_1/bidirectional_rnn/fw/fw/Rank+blstm_1/bidirectional_rnn/fw/fw/range/delta*

Tidx0*
_output_shapes
:
А
/blstm_1/bidirectional_rnn/fw/fw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
m
+blstm_1/bidirectional_rnn/fw/fw/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ё
&blstm_1/bidirectional_rnn/fw/fw/concatConcatV2/blstm_1/bidirectional_rnn/fw/fw/concat/values_0%blstm_1/bidirectional_rnn/fw/fw/range+blstm_1/bidirectional_rnn/fw/fw/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
│
)blstm_1/bidirectional_rnn/fw/fw/transpose	Transposeconcat&blstm_1/bidirectional_rnn/fw/fw/concat*5
_output_shapes#
!:                  А*
Tperm0*
T0
t
/blstm_1/bidirectional_rnn/fw/fw/sequence_lengthIdentityinputs/Placeholder_2*
_output_shapes
:*
T0
О
%blstm_1/bidirectional_rnn/fw/fw/ShapeShape)blstm_1/bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:
}
3blstm_1/bidirectional_rnn/fw/fw/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0

5blstm_1/bidirectional_rnn/fw/fw/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

5blstm_1/bidirectional_rnn/fw/fw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Щ
-blstm_1/bidirectional_rnn/fw/fw/strided_sliceStridedSlice%blstm_1/bidirectional_rnn/fw/fw/Shape3blstm_1/bidirectional_rnn/fw/fw/strided_slice/stack5blstm_1/bidirectional_rnn/fw/fw/strided_slice/stack_15blstm_1/bidirectional_rnn/fw/fw/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ъ
Xblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ь
Tblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims
ExpandDims-blstm_1/bidirectional_rnn/fw/fw/strided_sliceXblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
Ъ
Oblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:А
Ч
Ublstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ф
Pblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concatConcatV2Tblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDimsOblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ConstUblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
Ъ
Ublstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
┼
Oblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zerosFillPblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concatUblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros/Const*
T0*

index_type0*(
_output_shapes
:         А
Ь
Zblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
а
Vblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1
ExpandDims-blstm_1/bidirectional_rnn/fw/fw/strided_sliceZblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dim*
_output_shapes
:*

Tdim0*
T0
Ь
Qblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Const*
dtype0*
_output_shapes
:*
valueB:А
Ь
Zblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
а
Vblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2
ExpandDims-blstm_1/bidirectional_rnn/fw/fw/strided_sliceZblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0*
_output_shapes
:
Ь
Qblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Const*
valueB:А*
dtype0*
_output_shapes
:
Щ
Wblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ь
Rblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2Vblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2Qblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Wblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N
Ь
Wblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
╦
Qblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1FillRblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1Wblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/Const*(
_output_shapes
:         А*
T0*

index_type0
Ь
Zblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dimConst*
_output_shapes
: *
value	B : *
dtype0
а
Vblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3
ExpandDims-blstm_1/bidirectional_rnn/fw/fw/strided_sliceZblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dim*

Tdim0*
T0*
_output_shapes
:
Ь
Qblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/Const_3Const*
valueB:А*
dtype0*
_output_shapes
:
Я
'blstm_1/bidirectional_rnn/fw/fw/Shape_1Shape/blstm_1/bidirectional_rnn/fw/fw/sequence_length*
T0*
out_type0*#
_output_shapes
:         
Ц
%blstm_1/bidirectional_rnn/fw/fw/stackPack-blstm_1/bidirectional_rnn/fw/fw/strided_slice*
T0*

axis *
N*
_output_shapes
:
м
%blstm_1/bidirectional_rnn/fw/fw/EqualEqual'blstm_1/bidirectional_rnn/fw/fw/Shape_1%blstm_1/bidirectional_rnn/fw/fw/stack*
T0*#
_output_shapes
:         
o
%blstm_1/bidirectional_rnn/fw/fw/ConstConst*
valueB: *
dtype0*
_output_shapes
:
н
#blstm_1/bidirectional_rnn/fw/fw/AllAll%blstm_1/bidirectional_rnn/fw/fw/Equal%blstm_1/bidirectional_rnn/fw/fw/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
╝
,blstm_1/bidirectional_rnn/fw/fw/Assert/ConstConst*
dtype0*
_output_shapes
: *`
valueWBU BOExpected shape for Tensor blstm_1/bidirectional_rnn/fw/fw/sequence_length:0 is 

.blstm_1/bidirectional_rnn/fw/fw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
─
4blstm_1/bidirectional_rnn/fw/fw/Assert/Assert/data_0Const*`
valueWBU BOExpected shape for Tensor blstm_1/bidirectional_rnn/fw/fw/sequence_length:0 is *
dtype0*
_output_shapes
: 
Е
4blstm_1/bidirectional_rnn/fw/fw/Assert/Assert/data_2Const*
_output_shapes
: *!
valueB B but saw shape: *
dtype0
╕
-blstm_1/bidirectional_rnn/fw/fw/Assert/AssertAssert#blstm_1/bidirectional_rnn/fw/fw/All4blstm_1/bidirectional_rnn/fw/fw/Assert/Assert/data_0%blstm_1/bidirectional_rnn/fw/fw/stack4blstm_1/bidirectional_rnn/fw/fw/Assert/Assert/data_2'blstm_1/bidirectional_rnn/fw/fw/Shape_1*
T
2*
	summarize
╗
+blstm_1/bidirectional_rnn/fw/fw/CheckSeqLenIdentity/blstm_1/bidirectional_rnn/fw/fw/sequence_length.^blstm_1/bidirectional_rnn/fw/fw/Assert/Assert*
T0*
_output_shapes
:
Р
'blstm_1/bidirectional_rnn/fw/fw/Shape_2Shape)blstm_1/bidirectional_rnn/fw/fw/transpose*
_output_shapes
:*
T0*
out_type0

5blstm_1/bidirectional_rnn/fw/fw/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
Б
7blstm_1/bidirectional_rnn/fw/fw/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Б
7blstm_1/bidirectional_rnn/fw/fw/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
г
/blstm_1/bidirectional_rnn/fw/fw/strided_slice_1StridedSlice'blstm_1/bidirectional_rnn/fw/fw/Shape_25blstm_1/bidirectional_rnn/fw/fw/strided_slice_1/stack7blstm_1/bidirectional_rnn/fw/fw/strided_slice_1/stack_17blstm_1/bidirectional_rnn/fw/fw/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
Р
'blstm_1/bidirectional_rnn/fw/fw/Shape_3Shape)blstm_1/bidirectional_rnn/fw/fw/transpose*
_output_shapes
:*
T0*
out_type0

5blstm_1/bidirectional_rnn/fw/fw/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
Б
7blstm_1/bidirectional_rnn/fw/fw/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Б
7blstm_1/bidirectional_rnn/fw/fw/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
г
/blstm_1/bidirectional_rnn/fw/fw/strided_slice_2StridedSlice'blstm_1/bidirectional_rnn/fw/fw/Shape_35blstm_1/bidirectional_rnn/fw/fw/strided_slice_2/stack7blstm_1/bidirectional_rnn/fw/fw/strided_slice_2/stack_17blstm_1/bidirectional_rnn/fw/fw/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
p
.blstm_1/bidirectional_rnn/fw/fw/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
╩
*blstm_1/bidirectional_rnn/fw/fw/ExpandDims
ExpandDims/blstm_1/bidirectional_rnn/fw/fw/strided_slice_2.blstm_1/bidirectional_rnn/fw/fw/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
r
'blstm_1/bidirectional_rnn/fw/fw/Const_1Const*
dtype0*
_output_shapes
:*
valueB:А
o
-blstm_1/bidirectional_rnn/fw/fw/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Є
(blstm_1/bidirectional_rnn/fw/fw/concat_1ConcatV2*blstm_1/bidirectional_rnn/fw/fw/ExpandDims'blstm_1/bidirectional_rnn/fw/fw/Const_1-blstm_1/bidirectional_rnn/fw/fw/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
p
+blstm_1/bidirectional_rnn/fw/fw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╔
%blstm_1/bidirectional_rnn/fw/fw/zerosFill(blstm_1/bidirectional_rnn/fw/fw/concat_1+blstm_1/bidirectional_rnn/fw/fw/zeros/Const*(
_output_shapes
:         А*
T0*

index_type0
|
&blstm_1/bidirectional_rnn/fw/fw/Rank_1Rank+blstm_1/bidirectional_rnn/fw/fw/CheckSeqLen*
T0*
_output_shapes
: 
o
-blstm_1/bidirectional_rnn/fw/fw/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
o
-blstm_1/bidirectional_rnn/fw/fw/range_1/deltaConst*
dtype0*
_output_shapes
: *
value	B :
ч
'blstm_1/bidirectional_rnn/fw/fw/range_1Range-blstm_1/bidirectional_rnn/fw/fw/range_1/start&blstm_1/bidirectional_rnn/fw/fw/Rank_1-blstm_1/bidirectional_rnn/fw/fw/range_1/delta*#
_output_shapes
:         *

Tidx0
╛
#blstm_1/bidirectional_rnn/fw/fw/MinMin+blstm_1/bidirectional_rnn/fw/fw/CheckSeqLen'blstm_1/bidirectional_rnn/fw/fw/range_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
|
&blstm_1/bidirectional_rnn/fw/fw/Rank_2Rank+blstm_1/bidirectional_rnn/fw/fw/CheckSeqLen*
T0*
_output_shapes
: 
o
-blstm_1/bidirectional_rnn/fw/fw/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
o
-blstm_1/bidirectional_rnn/fw/fw/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ч
'blstm_1/bidirectional_rnn/fw/fw/range_2Range-blstm_1/bidirectional_rnn/fw/fw/range_2/start&blstm_1/bidirectional_rnn/fw/fw/Rank_2-blstm_1/bidirectional_rnn/fw/fw/range_2/delta*#
_output_shapes
:         *

Tidx0
╛
#blstm_1/bidirectional_rnn/fw/fw/MaxMax+blstm_1/bidirectional_rnn/fw/fw/CheckSeqLen'blstm_1/bidirectional_rnn/fw/fw/range_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
f
$blstm_1/bidirectional_rnn/fw/fw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
╪
+blstm_1/bidirectional_rnn/fw/fw/TensorArrayTensorArrayV3/blstm_1/bidirectional_rnn/fw/fw/strided_slice_1*K
tensor_array_name64blstm_1/bidirectional_rnn/fw/fw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *%
element_shape:         А*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
┘
-blstm_1/bidirectional_rnn/fw/fw/TensorArray_1TensorArrayV3/blstm_1/bidirectional_rnn/fw/fw/strided_slice_1*J
tensor_array_name53blstm_1/bidirectional_rnn/fw/fw/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *%
element_shape:         А*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
б
8blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeShape)blstm_1/bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:
Р
Fblstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Т
Hblstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Т
Hblstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
°
@blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_sliceStridedSlice8blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeFblstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackHblstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Hblstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
А
>blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
А
>blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
┤
8blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/rangeRange>blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/start@blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice>blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/delta*#
_output_shapes
:         *

Tidx0
Ц
Zblstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3-blstm_1/bidirectional_rnn/fw/fw/TensorArray_18blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/range)blstm_1/bidirectional_rnn/fw/fw/transpose/blstm_1/bidirectional_rnn/fw/fw/TensorArray_1:1*
T0*<
_class2
0.loc:@blstm_1/bidirectional_rnn/fw/fw/transpose*
_output_shapes
: 
k
)blstm_1/bidirectional_rnn/fw/fw/Maximum/xConst*
dtype0*
_output_shapes
: *
value	B :
г
'blstm_1/bidirectional_rnn/fw/fw/MaximumMaximum)blstm_1/bidirectional_rnn/fw/fw/Maximum/x#blstm_1/bidirectional_rnn/fw/fw/Max*
T0*
_output_shapes
: 
н
'blstm_1/bidirectional_rnn/fw/fw/MinimumMinimum/blstm_1/bidirectional_rnn/fw/fw/strided_slice_1'blstm_1/bidirectional_rnn/fw/fw/Maximum*
T0*
_output_shapes
: 
y
7blstm_1/bidirectional_rnn/fw/fw/while/iteration_counterConst*
dtype0*
_output_shapes
: *
value	B : 
Б
+blstm_1/bidirectional_rnn/fw/fw/while/EnterEnter7blstm_1/bidirectional_rnn/fw/fw/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
Ё
-blstm_1/bidirectional_rnn/fw/fw/while/Enter_1Enter$blstm_1/bidirectional_rnn/fw/fw/time*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
∙
-blstm_1/bidirectional_rnn/fw/fw/while/Enter_2Enter-blstm_1/bidirectional_rnn/fw/fw/TensorArray:1*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
н
-blstm_1/bidirectional_rnn/fw/fw/while/Enter_3EnterOblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*(
_output_shapes
:         А
п
-blstm_1/bidirectional_rnn/fw/fw/while/Enter_4EnterQblstm_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*(
_output_shapes
:         А
┬
+blstm_1/bidirectional_rnn/fw/fw/while/MergeMerge+blstm_1/bidirectional_rnn/fw/fw/while/Enter3blstm_1/bidirectional_rnn/fw/fw/while/NextIteration*
T0*
N*
_output_shapes
: : 
╚
-blstm_1/bidirectional_rnn/fw/fw/while/Merge_1Merge-blstm_1/bidirectional_rnn/fw/fw/while/Enter_15blstm_1/bidirectional_rnn/fw/fw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
╚
-blstm_1/bidirectional_rnn/fw/fw/while/Merge_2Merge-blstm_1/bidirectional_rnn/fw/fw/while/Enter_25blstm_1/bidirectional_rnn/fw/fw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
┌
-blstm_1/bidirectional_rnn/fw/fw/while/Merge_3Merge-blstm_1/bidirectional_rnn/fw/fw/while/Enter_35blstm_1/bidirectional_rnn/fw/fw/while/NextIteration_3*
T0*
N**
_output_shapes
:         А: 
┌
-blstm_1/bidirectional_rnn/fw/fw/while/Merge_4Merge-blstm_1/bidirectional_rnn/fw/fw/while/Enter_45blstm_1/bidirectional_rnn/fw/fw/while/NextIteration_4*
T0*
N**
_output_shapes
:         А: 
■
0blstm_1/bidirectional_rnn/fw/fw/while/Less/EnterEnter/blstm_1/bidirectional_rnn/fw/fw/strided_slice_1*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: *
T0*
is_constant(
▓
*blstm_1/bidirectional_rnn/fw/fw/while/LessLess+blstm_1/bidirectional_rnn/fw/fw/while/Merge0blstm_1/bidirectional_rnn/fw/fw/while/Less/Enter*
_output_shapes
: *
T0
°
2blstm_1/bidirectional_rnn/fw/fw/while/Less_1/EnterEnter'blstm_1/bidirectional_rnn/fw/fw/Minimum*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
╕
,blstm_1/bidirectional_rnn/fw/fw/while/Less_1Less-blstm_1/bidirectional_rnn/fw/fw/while/Merge_12blstm_1/bidirectional_rnn/fw/fw/while/Less_1/Enter*
_output_shapes
: *
T0
░
0blstm_1/bidirectional_rnn/fw/fw/while/LogicalAnd
LogicalAnd*blstm_1/bidirectional_rnn/fw/fw/while/Less,blstm_1/bidirectional_rnn/fw/fw/while/Less_1*
_output_shapes
: 
Д
.blstm_1/bidirectional_rnn/fw/fw/while/LoopCondLoopCond0blstm_1/bidirectional_rnn/fw/fw/while/LogicalAnd*
_output_shapes
: 
Ў
,blstm_1/bidirectional_rnn/fw/fw/while/SwitchSwitch+blstm_1/bidirectional_rnn/fw/fw/while/Merge.blstm_1/bidirectional_rnn/fw/fw/while/LoopCond*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/fw/while/Merge*
_output_shapes
: : 
№
.blstm_1/bidirectional_rnn/fw/fw/while/Switch_1Switch-blstm_1/bidirectional_rnn/fw/fw/while/Merge_1.blstm_1/bidirectional_rnn/fw/fw/while/LoopCond*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/fw/while/Merge_1*
_output_shapes
: : 
№
.blstm_1/bidirectional_rnn/fw/fw/while/Switch_2Switch-blstm_1/bidirectional_rnn/fw/fw/while/Merge_2.blstm_1/bidirectional_rnn/fw/fw/while/LoopCond*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/fw/while/Merge_2*
_output_shapes
: : 
а
.blstm_1/bidirectional_rnn/fw/fw/while/Switch_3Switch-blstm_1/bidirectional_rnn/fw/fw/while/Merge_3.blstm_1/bidirectional_rnn/fw/fw/while/LoopCond*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/fw/while/Merge_3*<
_output_shapes*
(:         А:         А
а
.blstm_1/bidirectional_rnn/fw/fw/while/Switch_4Switch-blstm_1/bidirectional_rnn/fw/fw/while/Merge_4.blstm_1/bidirectional_rnn/fw/fw/while/LoopCond*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/fw/while/Merge_4*<
_output_shapes*
(:         А:         А
Л
.blstm_1/bidirectional_rnn/fw/fw/while/IdentityIdentity.blstm_1/bidirectional_rnn/fw/fw/while/Switch:1*
T0*
_output_shapes
: 
П
0blstm_1/bidirectional_rnn/fw/fw/while/Identity_1Identity0blstm_1/bidirectional_rnn/fw/fw/while/Switch_1:1*
T0*
_output_shapes
: 
П
0blstm_1/bidirectional_rnn/fw/fw/while/Identity_2Identity0blstm_1/bidirectional_rnn/fw/fw/while/Switch_2:1*
T0*
_output_shapes
: 
б
0blstm_1/bidirectional_rnn/fw/fw/while/Identity_3Identity0blstm_1/bidirectional_rnn/fw/fw/while/Switch_3:1*
T0*(
_output_shapes
:         А
б
0blstm_1/bidirectional_rnn/fw/fw/while/Identity_4Identity0blstm_1/bidirectional_rnn/fw/fw/while/Switch_4:1*
T0*(
_output_shapes
:         А
Ю
+blstm_1/bidirectional_rnn/fw/fw/while/add/yConst/^blstm_1/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
о
)blstm_1/bidirectional_rnn/fw/fw/while/addAdd.blstm_1/bidirectional_rnn/fw/fw/while/Identity+blstm_1/bidirectional_rnn/fw/fw/while/add/y*
T0*
_output_shapes
: 
Н
=blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterEnter-blstm_1/bidirectional_rnn/fw/fw/TensorArray_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╕
?blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1EnterZblstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
╡
7blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3TensorArrayReadV3=blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter0blstm_1/bidirectional_rnn/fw/fw/while/Identity_1?blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:         А
Д
8blstm_1/bidirectional_rnn/fw/fw/while/GreaterEqual/EnterEnter+blstm_1/bidirectional_rnn/fw/fw/CheckSeqLen*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
╤
2blstm_1/bidirectional_rnn/fw/fw/while/GreaterEqualGreaterEqual0blstm_1/bidirectional_rnn/fw/fw/while/Identity_18blstm_1/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter*
T0*
_output_shapes
:
н
7blstm_1/bidirectional_rnn/fw/fw/while/dropout/keep_probConst/^blstm_1/bidirectional_rnn/fw/fw/while/Identity*
valueB
 *fff?*
dtype0*
_output_shapes
: 
к
3blstm_1/bidirectional_rnn/fw/fw/while/dropout/ShapeShape7blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
╢
@blstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/minConst/^blstm_1/bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *    
╢
@blstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/maxConst/^blstm_1/bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  А?
ъ
Jblstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniformRandomUniform3blstm_1/bidirectional_rnn/fw/fw/while/dropout/Shape*
T0*
dtype0*
seed2Ю*(
_output_shapes
:         А*

seed 
ь
@blstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/subSub@blstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/max@blstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/min*
T0*
_output_shapes
: 
И
@blstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/mulMulJblstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniform@blstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/sub*
T0*(
_output_shapes
:         А
·
<blstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniformAdd@blstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/mul@blstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/min*
T0*(
_output_shapes
:         А
т
1blstm_1/bidirectional_rnn/fw/fw/while/dropout/addAdd7blstm_1/bidirectional_rnn/fw/fw/while/dropout/keep_prob<blstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform*(
_output_shapes
:         А*
T0
в
3blstm_1/bidirectional_rnn/fw/fw/while/dropout/FloorFloor1blstm_1/bidirectional_rnn/fw/fw/while/dropout/add*
T0*(
_output_shapes
:         А
с
1blstm_1/bidirectional_rnn/fw/fw/while/dropout/divRealDiv7blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV37blstm_1/bidirectional_rnn/fw/fw/while/dropout/keep_prob*
T0*(
_output_shapes
:         А
╙
1blstm_1/bidirectional_rnn/fw/fw/while/dropout/mulMul1blstm_1/bidirectional_rnn/fw/fw/while/dropout/div3blstm_1/bidirectional_rnn/fw/fw/while/dropout/Floor*
T0*(
_output_shapes
:         А
с
Nblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
valueB"А     *
dtype0*
_output_shapes
:
╙
Lblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
valueB
 *bЧз╜*
dtype0*
_output_shapes
: 
╙
Lblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
valueB
 *bЧз=
╦
Vblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformNblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/shape*
seed2й*
dtype0* 
_output_shapes
:
АА*

seed *
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel
╥
Lblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/subSubLblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/maxLblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
_output_shapes
: 
ц
Lblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/mulMulVblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformLblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel
╪
Hblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniformAddLblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/mulLblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel* 
_output_shapes
:
АА
ч
-blstm_1/bidirectional_rnn/fw/lstm_cell/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
	container *
shape:
АА
═
4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/AssignAssign-blstm_1/bidirectional_rnn/fw/lstm_cell/kernelHblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
Ш
2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/readIdentity-blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
T0* 
_output_shapes
:
АА
╠
=blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zerosConst*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
┘
+blstm_1/bidirectional_rnn/fw/lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
	container *
shape:А
╖
2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/AssignAssign+blstm_1/bidirectional_rnn/fw/lstm_cell/bias=blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias
П
0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/readIdentity+blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
T0*
_output_shapes	
:А
о
;blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat/axisConst/^blstm_1/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
м
6blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concatConcatV21blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul0blstm_1/bidirectional_rnn/fw/fw/while/Identity_4;blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat/axis*
T0*
N*(
_output_shapes
:         А*

Tidx0
Ч
<blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/EnterEnter2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context* 
_output_shapes
:
АА
П
6blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMulMatMul6blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat<blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter*
T0*
transpose_a( *(
_output_shapes
:         А*
transpose_b( 
С
=blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/EnterEnter0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes	
:А
Г
7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAddBiasAdd6blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul=blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:         А
и
5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/ConstConst/^blstm_1/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
▓
?blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split/split_dimConst/^blstm_1/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╕
5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/splitSplit?blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split/split_dim7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd*
T0*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А
л
5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add/yConst/^blstm_1/bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
▌
3blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/addAdd7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split:25blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add/y*(
_output_shapes
:         А*
T0
к
7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/SigmoidSigmoid3blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add*
T0*(
_output_shapes
:         А
╪
3blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mulMul7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid0blstm_1/bidirectional_rnn/fw/fw/while/Identity_3*
T0*(
_output_shapes
:         А
о
9blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1Sigmoid5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split*(
_output_shapes
:         А*
T0
и
4blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/TanhTanh7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split:1*(
_output_shapes
:         А*
T0
р
5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1Mul9blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_14blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh*(
_output_shapes
:         А*
T0
█
5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1Add3blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1*
T0*(
_output_shapes
:         А
░
9blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2Sigmoid7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split:3*
T0*(
_output_shapes
:         А
и
6blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1Tanh5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1*
T0*(
_output_shapes
:         А
т
5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2Mul9blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_26blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*
T0*(
_output_shapes
:         А
╥
2blstm_1/bidirectional_rnn/fw/fw/while/Select/EnterEnter%blstm_1/bidirectional_rnn/fw/fw/zeros*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*(
_output_shapes
:         А*C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context
╥
,blstm_1/bidirectional_rnn/fw/fw/while/SelectSelect2blstm_1/bidirectional_rnn/fw/fw/while/GreaterEqual2blstm_1/bidirectional_rnn/fw/fw/while/Select/Enter5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*(
_output_shapes
:         А
╥
.blstm_1/bidirectional_rnn/fw/fw/while/Select_1Select2blstm_1/bidirectional_rnn/fw/fw/while/GreaterEqual0blstm_1/bidirectional_rnn/fw/fw/while/Identity_35blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1*(
_output_shapes
:         А
╥
.blstm_1/bidirectional_rnn/fw/fw/while/Select_2Select2blstm_1/bidirectional_rnn/fw/fw/while/GreaterEqual0blstm_1/bidirectional_rnn/fw/fw/while/Identity_45blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*(
_output_shapes
:         А
ч
Oblstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter+blstm_1/bidirectional_rnn/fw/fw/TensorArray*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context
н
Iblstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Oblstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter0blstm_1/bidirectional_rnn/fw/fw/while/Identity_1,blstm_1/bidirectional_rnn/fw/fw/while/Select0blstm_1/bidirectional_rnn/fw/fw/while/Identity_2*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
_output_shapes
: 
а
-blstm_1/bidirectional_rnn/fw/fw/while/add_1/yConst/^blstm_1/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
┤
+blstm_1/bidirectional_rnn/fw/fw/while/add_1Add0blstm_1/bidirectional_rnn/fw/fw/while/Identity_1-blstm_1/bidirectional_rnn/fw/fw/while/add_1/y*
T0*
_output_shapes
: 
Р
3blstm_1/bidirectional_rnn/fw/fw/while/NextIterationNextIteration)blstm_1/bidirectional_rnn/fw/fw/while/add*
T0*
_output_shapes
: 
Ф
5blstm_1/bidirectional_rnn/fw/fw/while/NextIteration_1NextIteration+blstm_1/bidirectional_rnn/fw/fw/while/add_1*
_output_shapes
: *
T0
▓
5blstm_1/bidirectional_rnn/fw/fw/while/NextIteration_2NextIterationIblstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
й
5blstm_1/bidirectional_rnn/fw/fw/while/NextIteration_3NextIteration.blstm_1/bidirectional_rnn/fw/fw/while/Select_1*
T0*(
_output_shapes
:         А
й
5blstm_1/bidirectional_rnn/fw/fw/while/NextIteration_4NextIteration.blstm_1/bidirectional_rnn/fw/fw/while/Select_2*
T0*(
_output_shapes
:         А
Б
*blstm_1/bidirectional_rnn/fw/fw/while/ExitExit,blstm_1/bidirectional_rnn/fw/fw/while/Switch*
T0*
_output_shapes
: 
Е
,blstm_1/bidirectional_rnn/fw/fw/while/Exit_1Exit.blstm_1/bidirectional_rnn/fw/fw/while/Switch_1*
T0*
_output_shapes
: 
Е
,blstm_1/bidirectional_rnn/fw/fw/while/Exit_2Exit.blstm_1/bidirectional_rnn/fw/fw/while/Switch_2*
T0*
_output_shapes
: 
Ч
,blstm_1/bidirectional_rnn/fw/fw/while/Exit_3Exit.blstm_1/bidirectional_rnn/fw/fw/while/Switch_3*
T0*(
_output_shapes
:         А
Ч
,blstm_1/bidirectional_rnn/fw/fw/while/Exit_4Exit.blstm_1/bidirectional_rnn/fw/fw/while/Switch_4*
T0*(
_output_shapes
:         А
К
Bblstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3+blstm_1/bidirectional_rnn/fw/fw/TensorArray,blstm_1/bidirectional_rnn/fw/fw/while/Exit_2*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/fw/TensorArray*
_output_shapes
: 
╛
<blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/fw/TensorArray*
value	B : 
╛
<blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/fw/TensorArray*
value	B :
Ё
6blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/rangeRange<blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/range/startBblstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3<blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/range/delta*

Tidx0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/fw/TensorArray*#
_output_shapes
:         
Щ
Dblstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3+blstm_1/bidirectional_rnn/fw/fw/TensorArray6blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/range,blstm_1/bidirectional_rnn/fw/fw/while/Exit_2*%
element_shape:         А*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/fw/TensorArray*
dtype0*5
_output_shapes#
!:                  А
r
'blstm_1/bidirectional_rnn/fw/fw/Const_2Const*
dtype0*
_output_shapes
:*
valueB:А
h
&blstm_1/bidirectional_rnn/fw/fw/Rank_3Const*
value	B :*
dtype0*
_output_shapes
: 
o
-blstm_1/bidirectional_rnn/fw/fw/range_3/startConst*
value	B :*
dtype0*
_output_shapes
: 
o
-blstm_1/bidirectional_rnn/fw/fw/range_3/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
▐
'blstm_1/bidirectional_rnn/fw/fw/range_3Range-blstm_1/bidirectional_rnn/fw/fw/range_3/start&blstm_1/bidirectional_rnn/fw/fw/Rank_3-blstm_1/bidirectional_rnn/fw/fw/range_3/delta*

Tidx0*
_output_shapes
:
В
1blstm_1/bidirectional_rnn/fw/fw/concat_2/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
o
-blstm_1/bidirectional_rnn/fw/fw/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
∙
(blstm_1/bidirectional_rnn/fw/fw/concat_2ConcatV21blstm_1/bidirectional_rnn/fw/fw/concat_2/values_0'blstm_1/bidirectional_rnn/fw/fw/range_3-blstm_1/bidirectional_rnn/fw/fw/concat_2/axis*

Tidx0*
T0*
N*
_output_shapes
:
ї
+blstm_1/bidirectional_rnn/fw/fw/transpose_1	TransposeDblstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3(blstm_1/bidirectional_rnn/fw/fw/concat_2*
T0*5
_output_shapes#
!:                  А*
Tperm0
╔
,blstm_1/bidirectional_rnn/bw/ReverseSequenceReverseSequenceconcatinputs/Placeholder_2*
	batch_dim *
T0*
seq_dim*

Tlen0*5
_output_shapes#
!:                  А
f
$blstm_1/bidirectional_rnn/bw/bw/RankConst*
value	B :*
dtype0*
_output_shapes
: 
m
+blstm_1/bidirectional_rnn/bw/bw/range/startConst*
dtype0*
_output_shapes
: *
value	B :
m
+blstm_1/bidirectional_rnn/bw/bw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
╓
%blstm_1/bidirectional_rnn/bw/bw/rangeRange+blstm_1/bidirectional_rnn/bw/bw/range/start$blstm_1/bidirectional_rnn/bw/bw/Rank+blstm_1/bidirectional_rnn/bw/bw/range/delta*
_output_shapes
:*

Tidx0
А
/blstm_1/bidirectional_rnn/bw/bw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
m
+blstm_1/bidirectional_rnn/bw/bw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ё
&blstm_1/bidirectional_rnn/bw/bw/concatConcatV2/blstm_1/bidirectional_rnn/bw/bw/concat/values_0%blstm_1/bidirectional_rnn/bw/bw/range+blstm_1/bidirectional_rnn/bw/bw/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
┘
)blstm_1/bidirectional_rnn/bw/bw/transpose	Transpose,blstm_1/bidirectional_rnn/bw/ReverseSequence&blstm_1/bidirectional_rnn/bw/bw/concat*
Tperm0*
T0*5
_output_shapes#
!:                  А
t
/blstm_1/bidirectional_rnn/bw/bw/sequence_lengthIdentityinputs/Placeholder_2*
T0*
_output_shapes
:
О
%blstm_1/bidirectional_rnn/bw/bw/ShapeShape)blstm_1/bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:
}
3blstm_1/bidirectional_rnn/bw/bw/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:

5blstm_1/bidirectional_rnn/bw/bw/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5blstm_1/bidirectional_rnn/bw/bw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Щ
-blstm_1/bidirectional_rnn/bw/bw/strided_sliceStridedSlice%blstm_1/bidirectional_rnn/bw/bw/Shape3blstm_1/bidirectional_rnn/bw/bw/strided_slice/stack5blstm_1/bidirectional_rnn/bw/bw/strided_slice/stack_15blstm_1/bidirectional_rnn/bw/bw/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ъ
Xblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Ь
Tblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims
ExpandDims-blstm_1/bidirectional_rnn/bw/bw/strided_sliceXblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
Ъ
Oblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ConstConst*
valueB:А*
dtype0*
_output_shapes
:
Ч
Ublstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ф
Pblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concatConcatV2Tblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDimsOblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ConstUblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ъ
Ublstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
┼
Oblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zerosFillPblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concatUblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros/Const*
T0*

index_type0*(
_output_shapes
:         А
Ь
Zblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
а
Vblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1
ExpandDims-blstm_1/bidirectional_rnn/bw/bw/strided_sliceZblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dim*
_output_shapes
:*

Tdim0*
T0
Ь
Qblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Const*
valueB:А*
dtype0*
_output_shapes
:
Ь
Zblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
а
Vblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2
ExpandDims-blstm_1/bidirectional_rnn/bw/bw/strided_sliceZblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dim*
_output_shapes
:*

Tdim0*
T0
Ь
Qblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Const*
valueB:А*
dtype0*
_output_shapes
:
Щ
Wblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ь
Rblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2Vblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2Qblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Wblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
Ь
Wblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╦
Qblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1FillRblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1Wblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/Const*(
_output_shapes
:         А*
T0*

index_type0
Ь
Zblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
а
Vblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3
ExpandDims-blstm_1/bidirectional_rnn/bw/bw/strided_sliceZblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dim*
T0*
_output_shapes
:*

Tdim0
Ь
Qblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/Const_3Const*
dtype0*
_output_shapes
:*
valueB:А
Я
'blstm_1/bidirectional_rnn/bw/bw/Shape_1Shape/blstm_1/bidirectional_rnn/bw/bw/sequence_length*
T0*
out_type0*#
_output_shapes
:         
Ц
%blstm_1/bidirectional_rnn/bw/bw/stackPack-blstm_1/bidirectional_rnn/bw/bw/strided_slice*
T0*

axis *
N*
_output_shapes
:
м
%blstm_1/bidirectional_rnn/bw/bw/EqualEqual'blstm_1/bidirectional_rnn/bw/bw/Shape_1%blstm_1/bidirectional_rnn/bw/bw/stack*
T0*#
_output_shapes
:         
o
%blstm_1/bidirectional_rnn/bw/bw/ConstConst*
valueB: *
dtype0*
_output_shapes
:
н
#blstm_1/bidirectional_rnn/bw/bw/AllAll%blstm_1/bidirectional_rnn/bw/bw/Equal%blstm_1/bidirectional_rnn/bw/bw/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
╝
,blstm_1/bidirectional_rnn/bw/bw/Assert/ConstConst*`
valueWBU BOExpected shape for Tensor blstm_1/bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 

.blstm_1/bidirectional_rnn/bw/bw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
─
4blstm_1/bidirectional_rnn/bw/bw/Assert/Assert/data_0Const*
dtype0*
_output_shapes
: *`
valueWBU BOExpected shape for Tensor blstm_1/bidirectional_rnn/bw/bw/sequence_length:0 is 
Е
4blstm_1/bidirectional_rnn/bw/bw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
╕
-blstm_1/bidirectional_rnn/bw/bw/Assert/AssertAssert#blstm_1/bidirectional_rnn/bw/bw/All4blstm_1/bidirectional_rnn/bw/bw/Assert/Assert/data_0%blstm_1/bidirectional_rnn/bw/bw/stack4blstm_1/bidirectional_rnn/bw/bw/Assert/Assert/data_2'blstm_1/bidirectional_rnn/bw/bw/Shape_1*
T
2*
	summarize
╗
+blstm_1/bidirectional_rnn/bw/bw/CheckSeqLenIdentity/blstm_1/bidirectional_rnn/bw/bw/sequence_length.^blstm_1/bidirectional_rnn/bw/bw/Assert/Assert*
_output_shapes
:*
T0
Р
'blstm_1/bidirectional_rnn/bw/bw/Shape_2Shape)blstm_1/bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:

5blstm_1/bidirectional_rnn/bw/bw/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
Б
7blstm_1/bidirectional_rnn/bw/bw/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Б
7blstm_1/bidirectional_rnn/bw/bw/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
г
/blstm_1/bidirectional_rnn/bw/bw/strided_slice_1StridedSlice'blstm_1/bidirectional_rnn/bw/bw/Shape_25blstm_1/bidirectional_rnn/bw/bw/strided_slice_1/stack7blstm_1/bidirectional_rnn/bw/bw/strided_slice_1/stack_17blstm_1/bidirectional_rnn/bw/bw/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
Р
'blstm_1/bidirectional_rnn/bw/bw/Shape_3Shape)blstm_1/bidirectional_rnn/bw/bw/transpose*
_output_shapes
:*
T0*
out_type0

5blstm_1/bidirectional_rnn/bw/bw/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
Б
7blstm_1/bidirectional_rnn/bw/bw/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Б
7blstm_1/bidirectional_rnn/bw/bw/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
г
/blstm_1/bidirectional_rnn/bw/bw/strided_slice_2StridedSlice'blstm_1/bidirectional_rnn/bw/bw/Shape_35blstm_1/bidirectional_rnn/bw/bw/strided_slice_2/stack7blstm_1/bidirectional_rnn/bw/bw/strided_slice_2/stack_17blstm_1/bidirectional_rnn/bw/bw/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
p
.blstm_1/bidirectional_rnn/bw/bw/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
╩
*blstm_1/bidirectional_rnn/bw/bw/ExpandDims
ExpandDims/blstm_1/bidirectional_rnn/bw/bw/strided_slice_2.blstm_1/bidirectional_rnn/bw/bw/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
r
'blstm_1/bidirectional_rnn/bw/bw/Const_1Const*
valueB:А*
dtype0*
_output_shapes
:
o
-blstm_1/bidirectional_rnn/bw/bw/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Є
(blstm_1/bidirectional_rnn/bw/bw/concat_1ConcatV2*blstm_1/bidirectional_rnn/bw/bw/ExpandDims'blstm_1/bidirectional_rnn/bw/bw/Const_1-blstm_1/bidirectional_rnn/bw/bw/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
p
+blstm_1/bidirectional_rnn/bw/bw/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
╔
%blstm_1/bidirectional_rnn/bw/bw/zerosFill(blstm_1/bidirectional_rnn/bw/bw/concat_1+blstm_1/bidirectional_rnn/bw/bw/zeros/Const*(
_output_shapes
:         А*
T0*

index_type0
|
&blstm_1/bidirectional_rnn/bw/bw/Rank_1Rank+blstm_1/bidirectional_rnn/bw/bw/CheckSeqLen*
T0*
_output_shapes
: 
o
-blstm_1/bidirectional_rnn/bw/bw/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
o
-blstm_1/bidirectional_rnn/bw/bw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ч
'blstm_1/bidirectional_rnn/bw/bw/range_1Range-blstm_1/bidirectional_rnn/bw/bw/range_1/start&blstm_1/bidirectional_rnn/bw/bw/Rank_1-blstm_1/bidirectional_rnn/bw/bw/range_1/delta*

Tidx0*#
_output_shapes
:         
╛
#blstm_1/bidirectional_rnn/bw/bw/MinMin+blstm_1/bidirectional_rnn/bw/bw/CheckSeqLen'blstm_1/bidirectional_rnn/bw/bw/range_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
|
&blstm_1/bidirectional_rnn/bw/bw/Rank_2Rank+blstm_1/bidirectional_rnn/bw/bw/CheckSeqLen*
_output_shapes
: *
T0
o
-blstm_1/bidirectional_rnn/bw/bw/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
o
-blstm_1/bidirectional_rnn/bw/bw/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ч
'blstm_1/bidirectional_rnn/bw/bw/range_2Range-blstm_1/bidirectional_rnn/bw/bw/range_2/start&blstm_1/bidirectional_rnn/bw/bw/Rank_2-blstm_1/bidirectional_rnn/bw/bw/range_2/delta*

Tidx0*#
_output_shapes
:         
╛
#blstm_1/bidirectional_rnn/bw/bw/MaxMax+blstm_1/bidirectional_rnn/bw/bw/CheckSeqLen'blstm_1/bidirectional_rnn/bw/bw/range_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
f
$blstm_1/bidirectional_rnn/bw/bw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
╪
+blstm_1/bidirectional_rnn/bw/bw/TensorArrayTensorArrayV3/blstm_1/bidirectional_rnn/bw/bw/strided_slice_1*%
element_shape:         А*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*K
tensor_array_name64blstm_1/bidirectional_rnn/bw/bw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 
┘
-blstm_1/bidirectional_rnn/bw/bw/TensorArray_1TensorArrayV3/blstm_1/bidirectional_rnn/bw/bw/strided_slice_1*
_output_shapes

:: *%
element_shape:         А*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*J
tensor_array_name53blstm_1/bidirectional_rnn/bw/bw/dynamic_rnn/input_0*
dtype0
б
8blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeShape)blstm_1/bidirectional_rnn/bw/bw/transpose*
out_type0*
_output_shapes
:*
T0
Р
Fblstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
Т
Hblstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Т
Hblstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
°
@blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_sliceStridedSlice8blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeFblstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackHblstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Hblstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
А
>blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
А
>blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
┤
8blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/rangeRange>blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/start@blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice>blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/delta*#
_output_shapes
:         *

Tidx0
Ц
Zblstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3-blstm_1/bidirectional_rnn/bw/bw/TensorArray_18blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/range)blstm_1/bidirectional_rnn/bw/bw/transpose/blstm_1/bidirectional_rnn/bw/bw/TensorArray_1:1*
_output_shapes
: *
T0*<
_class2
0.loc:@blstm_1/bidirectional_rnn/bw/bw/transpose
k
)blstm_1/bidirectional_rnn/bw/bw/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
г
'blstm_1/bidirectional_rnn/bw/bw/MaximumMaximum)blstm_1/bidirectional_rnn/bw/bw/Maximum/x#blstm_1/bidirectional_rnn/bw/bw/Max*
T0*
_output_shapes
: 
н
'blstm_1/bidirectional_rnn/bw/bw/MinimumMinimum/blstm_1/bidirectional_rnn/bw/bw/strided_slice_1'blstm_1/bidirectional_rnn/bw/bw/Maximum*
T0*
_output_shapes
: 
y
7blstm_1/bidirectional_rnn/bw/bw/while/iteration_counterConst*
dtype0*
_output_shapes
: *
value	B : 
Б
+blstm_1/bidirectional_rnn/bw/bw/while/EnterEnter7blstm_1/bidirectional_rnn/bw/bw/while/iteration_counter*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0*
is_constant( 
Ё
-blstm_1/bidirectional_rnn/bw/bw/while/Enter_1Enter$blstm_1/bidirectional_rnn/bw/bw/time*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0*
is_constant( 
∙
-blstm_1/bidirectional_rnn/bw/bw/while/Enter_2Enter-blstm_1/bidirectional_rnn/bw/bw/TensorArray:1*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0*
is_constant( 
н
-blstm_1/bidirectional_rnn/bw/bw/while/Enter_3EnterOblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*(
_output_shapes
:         А
п
-blstm_1/bidirectional_rnn/bw/bw/while/Enter_4EnterQblstm_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*(
_output_shapes
:         А
┬
+blstm_1/bidirectional_rnn/bw/bw/while/MergeMerge+blstm_1/bidirectional_rnn/bw/bw/while/Enter3blstm_1/bidirectional_rnn/bw/bw/while/NextIteration*
N*
_output_shapes
: : *
T0
╚
-blstm_1/bidirectional_rnn/bw/bw/while/Merge_1Merge-blstm_1/bidirectional_rnn/bw/bw/while/Enter_15blstm_1/bidirectional_rnn/bw/bw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
╚
-blstm_1/bidirectional_rnn/bw/bw/while/Merge_2Merge-blstm_1/bidirectional_rnn/bw/bw/while/Enter_25blstm_1/bidirectional_rnn/bw/bw/while/NextIteration_2*
N*
_output_shapes
: : *
T0
┌
-blstm_1/bidirectional_rnn/bw/bw/while/Merge_3Merge-blstm_1/bidirectional_rnn/bw/bw/while/Enter_35blstm_1/bidirectional_rnn/bw/bw/while/NextIteration_3*
N**
_output_shapes
:         А: *
T0
┌
-blstm_1/bidirectional_rnn/bw/bw/while/Merge_4Merge-blstm_1/bidirectional_rnn/bw/bw/while/Enter_45blstm_1/bidirectional_rnn/bw/bw/while/NextIteration_4*
T0*
N**
_output_shapes
:         А: 
■
0blstm_1/bidirectional_rnn/bw/bw/while/Less/EnterEnter/blstm_1/bidirectional_rnn/bw/bw/strided_slice_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
▓
*blstm_1/bidirectional_rnn/bw/bw/while/LessLess+blstm_1/bidirectional_rnn/bw/bw/while/Merge0blstm_1/bidirectional_rnn/bw/bw/while/Less/Enter*
T0*
_output_shapes
: 
°
2blstm_1/bidirectional_rnn/bw/bw/while/Less_1/EnterEnter'blstm_1/bidirectional_rnn/bw/bw/Minimum*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0*
is_constant(
╕
,blstm_1/bidirectional_rnn/bw/bw/while/Less_1Less-blstm_1/bidirectional_rnn/bw/bw/while/Merge_12blstm_1/bidirectional_rnn/bw/bw/while/Less_1/Enter*
_output_shapes
: *
T0
░
0blstm_1/bidirectional_rnn/bw/bw/while/LogicalAnd
LogicalAnd*blstm_1/bidirectional_rnn/bw/bw/while/Less,blstm_1/bidirectional_rnn/bw/bw/while/Less_1*
_output_shapes
: 
Д
.blstm_1/bidirectional_rnn/bw/bw/while/LoopCondLoopCond0blstm_1/bidirectional_rnn/bw/bw/while/LogicalAnd*
_output_shapes
: 
Ў
,blstm_1/bidirectional_rnn/bw/bw/while/SwitchSwitch+blstm_1/bidirectional_rnn/bw/bw/while/Merge.blstm_1/bidirectional_rnn/bw/bw/while/LoopCond*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/bw/while/Merge*
_output_shapes
: : 
№
.blstm_1/bidirectional_rnn/bw/bw/while/Switch_1Switch-blstm_1/bidirectional_rnn/bw/bw/while/Merge_1.blstm_1/bidirectional_rnn/bw/bw/while/LoopCond*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/bw/while/Merge_1*
_output_shapes
: : 
№
.blstm_1/bidirectional_rnn/bw/bw/while/Switch_2Switch-blstm_1/bidirectional_rnn/bw/bw/while/Merge_2.blstm_1/bidirectional_rnn/bw/bw/while/LoopCond*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/bw/while/Merge_2*
_output_shapes
: : 
а
.blstm_1/bidirectional_rnn/bw/bw/while/Switch_3Switch-blstm_1/bidirectional_rnn/bw/bw/while/Merge_3.blstm_1/bidirectional_rnn/bw/bw/while/LoopCond*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/bw/while/Merge_3*<
_output_shapes*
(:         А:         А
а
.blstm_1/bidirectional_rnn/bw/bw/while/Switch_4Switch-blstm_1/bidirectional_rnn/bw/bw/while/Merge_4.blstm_1/bidirectional_rnn/bw/bw/while/LoopCond*<
_output_shapes*
(:         А:         А*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/bw/while/Merge_4
Л
.blstm_1/bidirectional_rnn/bw/bw/while/IdentityIdentity.blstm_1/bidirectional_rnn/bw/bw/while/Switch:1*
_output_shapes
: *
T0
П
0blstm_1/bidirectional_rnn/bw/bw/while/Identity_1Identity0blstm_1/bidirectional_rnn/bw/bw/while/Switch_1:1*
T0*
_output_shapes
: 
П
0blstm_1/bidirectional_rnn/bw/bw/while/Identity_2Identity0blstm_1/bidirectional_rnn/bw/bw/while/Switch_2:1*
T0*
_output_shapes
: 
б
0blstm_1/bidirectional_rnn/bw/bw/while/Identity_3Identity0blstm_1/bidirectional_rnn/bw/bw/while/Switch_3:1*
T0*(
_output_shapes
:         А
б
0blstm_1/bidirectional_rnn/bw/bw/while/Identity_4Identity0blstm_1/bidirectional_rnn/bw/bw/while/Switch_4:1*(
_output_shapes
:         А*
T0
Ю
+blstm_1/bidirectional_rnn/bw/bw/while/add/yConst/^blstm_1/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
о
)blstm_1/bidirectional_rnn/bw/bw/while/addAdd.blstm_1/bidirectional_rnn/bw/bw/while/Identity+blstm_1/bidirectional_rnn/bw/bw/while/add/y*
T0*
_output_shapes
: 
Н
=blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterEnter-blstm_1/bidirectional_rnn/bw/bw/TensorArray_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
╕
?blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1EnterZblstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
╡
7blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3TensorArrayReadV3=blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter0blstm_1/bidirectional_rnn/bw/bw/while/Identity_1?blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1*(
_output_shapes
:         А*
dtype0
Д
8blstm_1/bidirectional_rnn/bw/bw/while/GreaterEqual/EnterEnter+blstm_1/bidirectional_rnn/bw/bw/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
╤
2blstm_1/bidirectional_rnn/bw/bw/while/GreaterEqualGreaterEqual0blstm_1/bidirectional_rnn/bw/bw/while/Identity_18blstm_1/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter*
_output_shapes
:*
T0
н
7blstm_1/bidirectional_rnn/bw/bw/while/dropout/keep_probConst/^blstm_1/bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
: *
valueB
 *fff?*
dtype0
к
3blstm_1/bidirectional_rnn/bw/bw/while/dropout/ShapeShape7blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
╢
@blstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/minConst/^blstm_1/bidirectional_rnn/bw/bw/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
╢
@blstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/maxConst/^blstm_1/bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
: *
valueB
 *  А?*
dtype0
ъ
Jblstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniformRandomUniform3blstm_1/bidirectional_rnn/bw/bw/while/dropout/Shape*
T0*
dtype0*
seed2ц*(
_output_shapes
:         А*

seed 
ь
@blstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/subSub@blstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/max@blstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/min*
_output_shapes
: *
T0
И
@blstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/mulMulJblstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniform@blstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/sub*(
_output_shapes
:         А*
T0
·
<blstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniformAdd@blstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/mul@blstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/min*
T0*(
_output_shapes
:         А
т
1blstm_1/bidirectional_rnn/bw/bw/while/dropout/addAdd7blstm_1/bidirectional_rnn/bw/bw/while/dropout/keep_prob<blstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform*
T0*(
_output_shapes
:         А
в
3blstm_1/bidirectional_rnn/bw/bw/while/dropout/FloorFloor1blstm_1/bidirectional_rnn/bw/bw/while/dropout/add*
T0*(
_output_shapes
:         А
с
1blstm_1/bidirectional_rnn/bw/bw/while/dropout/divRealDiv7blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV37blstm_1/bidirectional_rnn/bw/bw/while/dropout/keep_prob*
T0*(
_output_shapes
:         А
╙
1blstm_1/bidirectional_rnn/bw/bw/while/dropout/mulMul1blstm_1/bidirectional_rnn/bw/bw/while/dropout/div3blstm_1/bidirectional_rnn/bw/bw/while/dropout/Floor*
T0*(
_output_shapes
:         А
с
Nblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
valueB"А     
╙
Lblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
valueB
 *bЧз╜*
dtype0*
_output_shapes
: 
╙
Lblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
valueB
 *bЧз=*
dtype0
╦
Vblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformNblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
seed2ё*
dtype0* 
_output_shapes
:
АА*

seed 
╥
Lblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/subSubLblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/maxLblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
_output_shapes
: 
ц
Lblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/mulMulVblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformLblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel
╪
Hblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniformAddLblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/mulLblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
АА
ч
-blstm_1/bidirectional_rnn/bw/lstm_cell/kernel
VariableV2*
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
	container 
═
4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/AssignAssign-blstm_1/bidirectional_rnn/bw/lstm_cell/kernelHblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
Ш
2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/readIdentity-blstm_1/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
АА*
T0
╠
=blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zerosConst*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
┘
+blstm_1/bidirectional_rnn/bw/lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
	container *
shape:А
╖
2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/AssignAssign+blstm_1/bidirectional_rnn/bw/lstm_cell/bias=blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
П
0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/readIdentity+blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
T0*
_output_shapes	
:А
о
;blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat/axisConst/^blstm_1/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
м
6blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concatConcatV21blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul0blstm_1/bidirectional_rnn/bw/bw/while/Identity_4;blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat/axis*
N*(
_output_shapes
:         А*

Tidx0*
T0
Ч
<blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/EnterEnter2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context* 
_output_shapes
:
АА
П
6blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMulMatMul6blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat<blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter*
T0*
transpose_a( *(
_output_shapes
:         А*
transpose_b( 
С
=blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/EnterEnter0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes	
:А
Г
7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAddBiasAdd6blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul=blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:         А
и
5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/ConstConst/^blstm_1/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
▓
?blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split/split_dimConst/^blstm_1/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╕
5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/splitSplit?blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split/split_dim7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd*
T0*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А
л
5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add/yConst/^blstm_1/bidirectional_rnn/bw/bw/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
▌
3blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/addAdd7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split:25blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add/y*(
_output_shapes
:         А*
T0
к
7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/SigmoidSigmoid3blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add*
T0*(
_output_shapes
:         А
╪
3blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mulMul7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid0blstm_1/bidirectional_rnn/bw/bw/while/Identity_3*
T0*(
_output_shapes
:         А
о
9blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1Sigmoid5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split*
T0*(
_output_shapes
:         А
и
4blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/TanhTanh7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split:1*
T0*(
_output_shapes
:         А
р
5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1Mul9blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_14blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*
T0*(
_output_shapes
:         А
█
5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1Add3blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1*
T0*(
_output_shapes
:         А
░
9blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2Sigmoid7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split:3*
T0*(
_output_shapes
:         А
и
6blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1Tanh5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1*
T0*(
_output_shapes
:         А
т
5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2Mul9blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_26blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*
T0*(
_output_shapes
:         А
╥
2blstm_1/bidirectional_rnn/bw/bw/while/Select/EnterEnter%blstm_1/bidirectional_rnn/bw/bw/zeros*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*(
_output_shapes
:         А
╥
,blstm_1/bidirectional_rnn/bw/bw/while/SelectSelect2blstm_1/bidirectional_rnn/bw/bw/while/GreaterEqual2blstm_1/bidirectional_rnn/bw/bw/while/Select/Enter5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*(
_output_shapes
:         А
╥
.blstm_1/bidirectional_rnn/bw/bw/while/Select_1Select2blstm_1/bidirectional_rnn/bw/bw/while/GreaterEqual0blstm_1/bidirectional_rnn/bw/bw/while/Identity_35blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1*(
_output_shapes
:         А
╥
.blstm_1/bidirectional_rnn/bw/bw/while/Select_2Select2blstm_1/bidirectional_rnn/bw/bw/while/GreaterEqual0blstm_1/bidirectional_rnn/bw/bw/while/Identity_45blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*(
_output_shapes
:         А*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2
ч
Oblstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter+blstm_1/bidirectional_rnn/bw/bw/TensorArray*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context
н
Iblstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Oblstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter0blstm_1/bidirectional_rnn/bw/bw/while/Identity_1,blstm_1/bidirectional_rnn/bw/bw/while/Select0blstm_1/bidirectional_rnn/bw/bw/while/Identity_2*
_output_shapes
: *
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2
а
-blstm_1/bidirectional_rnn/bw/bw/while/add_1/yConst/^blstm_1/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
┤
+blstm_1/bidirectional_rnn/bw/bw/while/add_1Add0blstm_1/bidirectional_rnn/bw/bw/while/Identity_1-blstm_1/bidirectional_rnn/bw/bw/while/add_1/y*
_output_shapes
: *
T0
Р
3blstm_1/bidirectional_rnn/bw/bw/while/NextIterationNextIteration)blstm_1/bidirectional_rnn/bw/bw/while/add*
T0*
_output_shapes
: 
Ф
5blstm_1/bidirectional_rnn/bw/bw/while/NextIteration_1NextIteration+blstm_1/bidirectional_rnn/bw/bw/while/add_1*
_output_shapes
: *
T0
▓
5blstm_1/bidirectional_rnn/bw/bw/while/NextIteration_2NextIterationIblstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
й
5blstm_1/bidirectional_rnn/bw/bw/while/NextIteration_3NextIteration.blstm_1/bidirectional_rnn/bw/bw/while/Select_1*(
_output_shapes
:         А*
T0
й
5blstm_1/bidirectional_rnn/bw/bw/while/NextIteration_4NextIteration.blstm_1/bidirectional_rnn/bw/bw/while/Select_2*
T0*(
_output_shapes
:         А
Б
*blstm_1/bidirectional_rnn/bw/bw/while/ExitExit,blstm_1/bidirectional_rnn/bw/bw/while/Switch*
T0*
_output_shapes
: 
Е
,blstm_1/bidirectional_rnn/bw/bw/while/Exit_1Exit.blstm_1/bidirectional_rnn/bw/bw/while/Switch_1*
T0*
_output_shapes
: 
Е
,blstm_1/bidirectional_rnn/bw/bw/while/Exit_2Exit.blstm_1/bidirectional_rnn/bw/bw/while/Switch_2*
T0*
_output_shapes
: 
Ч
,blstm_1/bidirectional_rnn/bw/bw/while/Exit_3Exit.blstm_1/bidirectional_rnn/bw/bw/while/Switch_3*
T0*(
_output_shapes
:         А
Ч
,blstm_1/bidirectional_rnn/bw/bw/while/Exit_4Exit.blstm_1/bidirectional_rnn/bw/bw/while/Switch_4*
T0*(
_output_shapes
:         А
К
Bblstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3+blstm_1/bidirectional_rnn/bw/bw/TensorArray,blstm_1/bidirectional_rnn/bw/bw/while/Exit_2*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
: 
╛
<blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/range/startConst*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/bw/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
╛
<blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/range/deltaConst*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/bw/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
Ё
6blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/rangeRange<blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/range/startBblstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3<blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/range/delta*#
_output_shapes
:         *

Tidx0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/bw/TensorArray
Щ
Dblstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3+blstm_1/bidirectional_rnn/bw/bw/TensorArray6blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/range,blstm_1/bidirectional_rnn/bw/bw/while/Exit_2*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/bw/TensorArray*
dtype0*5
_output_shapes#
!:                  А*%
element_shape:         А
r
'blstm_1/bidirectional_rnn/bw/bw/Const_2Const*
valueB:А*
dtype0*
_output_shapes
:
h
&blstm_1/bidirectional_rnn/bw/bw/Rank_3Const*
dtype0*
_output_shapes
: *
value	B :
o
-blstm_1/bidirectional_rnn/bw/bw/range_3/startConst*
value	B :*
dtype0*
_output_shapes
: 
o
-blstm_1/bidirectional_rnn/bw/bw/range_3/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
▐
'blstm_1/bidirectional_rnn/bw/bw/range_3Range-blstm_1/bidirectional_rnn/bw/bw/range_3/start&blstm_1/bidirectional_rnn/bw/bw/Rank_3-blstm_1/bidirectional_rnn/bw/bw/range_3/delta*
_output_shapes
:*

Tidx0
В
1blstm_1/bidirectional_rnn/bw/bw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
o
-blstm_1/bidirectional_rnn/bw/bw/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
∙
(blstm_1/bidirectional_rnn/bw/bw/concat_2ConcatV21blstm_1/bidirectional_rnn/bw/bw/concat_2/values_0'blstm_1/bidirectional_rnn/bw/bw/range_3-blstm_1/bidirectional_rnn/bw/bw/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
ї
+blstm_1/bidirectional_rnn/bw/bw/transpose_1	TransposeDblstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3(blstm_1/bidirectional_rnn/bw/bw/concat_2*
T0*5
_output_shapes#
!:                  А*
Tperm0
┘
blstm_1/ReverseSequenceReverseSequence+blstm_1/bidirectional_rnn/bw/bw/transpose_1inputs/Placeholder_2*
	batch_dim *
T0*
seq_dim*

Tlen0*5
_output_shapes#
!:                  А
O
concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
╛
concat_1ConcatV2+blstm_1/bidirectional_rnn/fw/fw/transpose_1blstm_1/ReverseSequenceconcat_1/axis*

Tidx0*
T0*
N*5
_output_shapes#
!:                  А
p
ShapeShape+blstm_1/bidirectional_rnn/fw/fw/transpose_1*
T0*
out_type0*
_output_shapes
:
]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
∙
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
M
range/startConst*
value	B : *
dtype0*
_output_shapes
: 
M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
h
rangeRangerange/startstrided_slicerange/delta*#
_output_shapes
:         *

Tidx0
G
sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
J
subSubinputs/Placeholder_2sub/y*
T0*
_output_shapes
:
`
stackPackrangesub*
T0*

axis*
N*'
_output_shapes
:         
С
GatherNdGatherNd+blstm_1/bidirectional_rnn/fw/fw/transpose_1stack*(
_output_shapes
:         А*
Tindices0*
Tparams0


GatherNd_1GatherNdblstm_1/ReverseSequencestack*
Tindices0*
Tparams0*(
_output_shapes
:         А
O
concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B :
Б
concat_2ConcatV2GatherNd
GatherNd_1concat_2/axis*
N*(
_output_shapes
:         А*

Tidx0*
T0
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"   @   
С
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *М7╛*
dtype0*
_output_shapes
: 
С
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *М7>*
dtype0*
_output_shapes
: 
ч
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А@*

seed *
T0*
_class
loc:@dense/kernel*
seed2┼
╬
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@dense/kernel
с
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	А@
╙
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	А@
г
dense/kernel
VariableV2*
dtype0*
_output_shapes
:	А@*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	А@
╚
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	А@
v
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	А@
И
dense/bias/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Х

dense/bias
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@dense/bias
▓
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:@
k
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes
:@
С
dense/dense/MatMulMatMulconcat_2dense/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:         @
М
dense/dense/BiasAddBiasAdddense/dense/MatMuldense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         @
_
dense/dense/TanhTanhdense/dense/BiasAdd*
T0*'
_output_shapes
:         @
г
/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@dense_1/kernel*
valueB"@      
Х
-dense_1/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_1/kernel*
valueB
 *┴╓Ф╛*
dtype0*
_output_shapes
: 
Х
-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
valueB
 *┴╓Ф>
ь
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed *
T0*!
_class
loc:@dense_1/kernel*
seed2╓
╓
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_1/kernel
ш
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*!
_class
loc:@dense_1/kernel
┌
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
е
dense_1/kernel
VariableV2*!
_class
loc:@dense_1/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
╧
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:@
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
М
dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@dense_1/bias*
valueB*    
Щ
dense_1/bias
VariableV2*
_class
loc:@dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
║
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
q
dense_1/bias/readIdentitydense_1/bias*
_output_shapes
:*
T0*
_class
loc:@dense_1/bias
Э
dense/dense_1/MatMulMatMuldense/dense/Tanhdense_1/kernel/read*
transpose_a( *'
_output_shapes
:         *
transpose_b( *
T0
Т
dense/dense_1/BiasAddBiasAdddense/dense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
i
dense/dense_1/SoftmaxSoftmaxdense/dense_1/BiasAdd*'
_output_shapes
:         *
T0
m
+loss/softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
,loss/softmax_cross_entropy_with_logits/ShapeShapedense/dense_1/Softmax*
T0*
out_type0*
_output_shapes
:
o
-loss/softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
Г
.loss/softmax_cross_entropy_with_logits/Shape_1Shapedense/dense_1/Softmax*
T0*
out_type0*
_output_shapes
:
n
,loss/softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
п
*loss/softmax_cross_entropy_with_logits/SubSub-loss/softmax_cross_entropy_with_logits/Rank_1,loss/softmax_cross_entropy_with_logits/Sub/y*
_output_shapes
: *
T0
а
2loss/softmax_cross_entropy_with_logits/Slice/beginPack*loss/softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N*
_output_shapes
:
{
1loss/softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
■
,loss/softmax_cross_entropy_with_logits/SliceSlice.loss/softmax_cross_entropy_with_logits/Shape_12loss/softmax_cross_entropy_with_logits/Slice/begin1loss/softmax_cross_entropy_with_logits/Slice/size*
_output_shapes
:*
Index0*
T0
Й
6loss/softmax_cross_entropy_with_logits/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
         
t
2loss/softmax_cross_entropy_with_logits/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Н
-loss/softmax_cross_entropy_with_logits/concatConcatV26loss/softmax_cross_entropy_with_logits/concat/values_0,loss/softmax_cross_entropy_with_logits/Slice2loss/softmax_cross_entropy_with_logits/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
╚
.loss/softmax_cross_entropy_with_logits/ReshapeReshapedense/dense_1/Softmax-loss/softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*0
_output_shapes
:                  
o
-loss/softmax_cross_entropy_with_logits/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
В
.loss/softmax_cross_entropy_with_logits/Shape_2Shapeinputs/Placeholder_1*
T0*
out_type0*
_output_shapes
:
p
.loss/softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
│
,loss/softmax_cross_entropy_with_logits/Sub_1Sub-loss/softmax_cross_entropy_with_logits/Rank_2.loss/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
д
4loss/softmax_cross_entropy_with_logits/Slice_1/beginPack,loss/softmax_cross_entropy_with_logits/Sub_1*
N*
_output_shapes
:*
T0*

axis 
}
3loss/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Д
.loss/softmax_cross_entropy_with_logits/Slice_1Slice.loss/softmax_cross_entropy_with_logits/Shape_24loss/softmax_cross_entropy_with_logits/Slice_1/begin3loss/softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:
Л
8loss/softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
v
4loss/softmax_cross_entropy_with_logits/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Х
/loss/softmax_cross_entropy_with_logits/concat_1ConcatV28loss/softmax_cross_entropy_with_logits/concat_1/values_0.loss/softmax_cross_entropy_with_logits/Slice_14loss/softmax_cross_entropy_with_logits/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
╦
0loss/softmax_cross_entropy_with_logits/Reshape_1Reshapeinputs/Placeholder_1/loss/softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*0
_output_shapes
:                  
є
&loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits.loss/softmax_cross_entropy_with_logits/Reshape0loss/softmax_cross_entropy_with_logits/Reshape_1*?
_output_shapes-
+:         :                  *
T0
p
.loss/softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
▒
,loss/softmax_cross_entropy_with_logits/Sub_2Sub+loss/softmax_cross_entropy_with_logits/Rank.loss/softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
~
4loss/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
г
3loss/softmax_cross_entropy_with_logits/Slice_2/sizePack,loss/softmax_cross_entropy_with_logits/Sub_2*
N*
_output_shapes
:*
T0*

axis 
В
.loss/softmax_cross_entropy_with_logits/Slice_2Slice,loss/softmax_cross_entropy_with_logits/Shape4loss/softmax_cross_entropy_with_logits/Slice_2/begin3loss/softmax_cross_entropy_with_logits/Slice_2/size*
Index0*
T0*
_output_shapes
:
╧
0loss/softmax_cross_entropy_with_logits/Reshape_2Reshape&loss/softmax_cross_entropy_with_logits.loss/softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0*#
_output_shapes
:         
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Н
	loss/MeanMean0loss/softmax_cross_entropy_with_logits/Reshape_2
loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
S
gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
├
gradients/f_count_1Entergradients/f_count*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: *
T0*
is_constant( 
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
T0*
N*
_output_shapes
: : 
~
gradients/SwitchSwitchgradients/Merge.blstm_1/bidirectional_rnn/fw/fw/while/LoopCond*
T0*
_output_shapes
: : 
В
gradients/Add/yConst/^blstm_1/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
T0*
_output_shapes
: 
N
gradients/f_count_2Exitgradients/Switch*
T0*
_output_shapes
: 
S
gradients/b_countConst*
dtype0*
_output_shapes
: *
value	B :
╧
gradients/b_count_1Entergradients/f_count_2*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
╓
gradients/GreaterEqual/EnterEntergradients/b_count*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
T0*
_output_shapes
: : 
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
P
gradients/b_count_3Exitgradients/Switch_1*
T0*
_output_shapes
: 
U
gradients/f_count_3Const*
value	B : *
dtype0*
_output_shapes
: 
┼
gradients/f_count_4Entergradients/f_count_3*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
v
gradients/Merge_2Mergegradients/f_count_4gradients/NextIteration_2*
T0*
N*
_output_shapes
: : 
В
gradients/Switch_2Switchgradients/Merge_2.blstm_1/bidirectional_rnn/bw/bw/while/LoopCond*
T0*
_output_shapes
: : 
Д
gradients/Add_1/yConst/^blstm_1/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
gradients/Add_1Addgradients/Switch_2:1gradients/Add_1/y*
T0*
_output_shapes
: 
P
gradients/f_count_5Exitgradients/Switch_2*
T0*
_output_shapes
: 
U
gradients/b_count_4Const*
value	B :*
dtype0*
_output_shapes
: 
╧
gradients/b_count_5Entergradients/f_count_5*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0*
is_constant( 
v
gradients/Merge_3Mergegradients/b_count_5gradients/NextIteration_3*
T0*
N*
_output_shapes
: : 
┌
gradients/GreaterEqual_1/EnterEntergradients/b_count_4*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0*
is_constant(
|
gradients/GreaterEqual_1GreaterEqualgradients/Merge_3gradients/GreaterEqual_1/Enter*
T0*
_output_shapes
: 
Q
gradients/b_count_6LoopCondgradients/GreaterEqual_1*
_output_shapes
: 
g
gradients/Switch_3Switchgradients/Merge_3gradients/b_count_6*
_output_shapes
: : *
T0
m
gradients/Sub_1Subgradients/Switch_3:1gradients/GreaterEqual_1/Enter*
_output_shapes
: *
T0
P
gradients/b_count_7Exitgradients/Switch_3*
_output_shapes
: *
T0
U
gradients/f_count_6Const*
value	B : *
dtype0*
_output_shapes
: 
┼
gradients/f_count_7Entergradients/f_count_6*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: *
T0*
is_constant( 
v
gradients/Merge_4Mergegradients/f_count_7gradients/NextIteration_4*
T0*
N*
_output_shapes
: : 
В
gradients/Switch_4Switchgradients/Merge_4.blstm_0/bidirectional_rnn/fw/fw/while/LoopCond*
T0*
_output_shapes
: : 
Д
gradients/Add_2/yConst/^blstm_0/bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
`
gradients/Add_2Addgradients/Switch_4:1gradients/Add_2/y*
T0*
_output_shapes
: 
P
gradients/f_count_8Exitgradients/Switch_4*
_output_shapes
: *
T0
U
gradients/b_count_8Const*
value	B :*
dtype0*
_output_shapes
: 
╧
gradients/b_count_9Entergradients/f_count_8*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: *
T0*
is_constant( 
v
gradients/Merge_5Mergegradients/b_count_9gradients/NextIteration_5*
N*
_output_shapes
: : *
T0
┌
gradients/GreaterEqual_2/EnterEntergradients/b_count_8*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
|
gradients/GreaterEqual_2GreaterEqualgradients/Merge_5gradients/GreaterEqual_2/Enter*
T0*
_output_shapes
: 
R
gradients/b_count_10LoopCondgradients/GreaterEqual_2*
_output_shapes
: 
h
gradients/Switch_5Switchgradients/Merge_5gradients/b_count_10*
T0*
_output_shapes
: : 
m
gradients/Sub_2Subgradients/Switch_5:1gradients/GreaterEqual_2/Enter*
T0*
_output_shapes
: 
Q
gradients/b_count_11Exitgradients/Switch_5*
T0*
_output_shapes
: 
U
gradients/f_count_9Const*
dtype0*
_output_shapes
: *
value	B : 
╞
gradients/f_count_10Entergradients/f_count_9*
T0*
is_constant( *
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
w
gradients/Merge_6Mergegradients/f_count_10gradients/NextIteration_6*
T0*
N*
_output_shapes
: : 
В
gradients/Switch_6Switchgradients/Merge_6.blstm_0/bidirectional_rnn/bw/bw/while/LoopCond*
T0*
_output_shapes
: : 
Д
gradients/Add_3/yConst/^blstm_0/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
gradients/Add_3Addgradients/Switch_6:1gradients/Add_3/y*
T0*
_output_shapes
: 
Q
gradients/f_count_11Exitgradients/Switch_6*
T0*
_output_shapes
: 
V
gradients/b_count_12Const*
value	B :*
dtype0*
_output_shapes
: 
╤
gradients/b_count_13Entergradients/f_count_11*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
w
gradients/Merge_7Mergegradients/b_count_13gradients/NextIteration_7*
T0*
N*
_output_shapes
: : 
█
gradients/GreaterEqual_3/EnterEntergradients/b_count_12*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
|
gradients/GreaterEqual_3GreaterEqualgradients/Merge_7gradients/GreaterEqual_3/Enter*
T0*
_output_shapes
: 
R
gradients/b_count_14LoopCondgradients/GreaterEqual_3*
_output_shapes
: 
h
gradients/Switch_7Switchgradients/Merge_7gradients/b_count_14*
_output_shapes
: : *
T0
m
gradients/Sub_3Subgradients/Switch_7:1gradients/GreaterEqual_3/Enter*
T0*
_output_shapes
: 
Q
gradients/b_count_15Exitgradients/Switch_7*
T0*
_output_shapes
: 
p
&gradients/loss/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ц
 gradients/loss/Mean_grad/ReshapeReshapegradients/Fill&gradients/loss/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
О
gradients/loss/Mean_grad/ShapeShape0loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*
_output_shapes
:
з
gradients/loss/Mean_grad/TileTile gradients/loss/Mean_grad/Reshapegradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
Р
 gradients/loss/Mean_grad/Shape_1Shape0loss/softmax_cross_entropy_with_logits/Reshape_2*
_output_shapes
:*
T0*
out_type0
c
 gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
h
gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
е
gradients/loss/Mean_grad/ProdProd gradients/loss/Mean_grad/Shape_1gradients/loss/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
 gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
й
gradients/loss/Mean_grad/Prod_1Prod gradients/loss/Mean_grad/Shape_2 gradients/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
d
"gradients/loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
С
 gradients/loss/Mean_grad/MaximumMaximumgradients/loss/Mean_grad/Prod_1"gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
П
!gradients/loss/Mean_grad/floordivFloorDivgradients/loss/Mean_grad/Prod gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
И
gradients/loss/Mean_grad/CastCast!gradients/loss/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
Ч
 gradients/loss/Mean_grad/truedivRealDivgradients/loss/Mean_grad/Tilegradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:         
л
Egradients/loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape&loss/softmax_cross_entropy_with_logits*
T0*
out_type0*
_output_shapes
:
ў
Ggradients/loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshape gradients/loss/Mean_grad/truedivEgradients/loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*#
_output_shapes
:         *
T0*
Tshape0
Ж
gradients/zeros_like	ZerosLike(loss/softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:                  
П
Dgradients/loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
         
Ы
@gradients/loss/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsGgradients/loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeDgradients/loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
ч
9gradients/loss/softmax_cross_entropy_with_logits_grad/mulMul@gradients/loss/softmax_cross_entropy_with_logits_grad/ExpandDims(loss/softmax_cross_entropy_with_logits:1*0
_output_shapes
:                  *
T0
╣
@gradients/loss/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax.loss/softmax_cross_entropy_with_logits/Reshape*
T0*0
_output_shapes
:                  
╜
9gradients/loss/softmax_cross_entropy_with_logits_grad/NegNeg@gradients/loss/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0*0
_output_shapes
:                  
С
Fgradients/loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
Я
Bgradients/loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsGgradients/loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeFgradients/loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:         
№
;gradients/loss/softmax_cross_entropy_with_logits_grad/mul_1MulBgradients/loss/softmax_cross_entropy_with_logits_grad/ExpandDims_19gradients/loss/softmax_cross_entropy_with_logits_grad/Neg*0
_output_shapes
:                  *
T0
╚
Fgradients/loss/softmax_cross_entropy_with_logits_grad/tuple/group_depsNoOp:^gradients/loss/softmax_cross_entropy_with_logits_grad/mul<^gradients/loss/softmax_cross_entropy_with_logits_grad/mul_1
ч
Ngradients/loss/softmax_cross_entropy_with_logits_grad/tuple/control_dependencyIdentity9gradients/loss/softmax_cross_entropy_with_logits_grad/mulG^gradients/loss/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/loss/softmax_cross_entropy_with_logits_grad/mul*0
_output_shapes
:                  
э
Pgradients/loss/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1Identity;gradients/loss/softmax_cross_entropy_with_logits_grad/mul_1G^gradients/loss/softmax_cross_entropy_with_logits_grad/tuple/group_deps*0
_output_shapes
:                  *
T0*N
_classD
B@loc:@gradients/loss/softmax_cross_entropy_with_logits_grad/mul_1
Ш
Cgradients/loss/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapedense/dense_1/Softmax*
T0*
out_type0*
_output_shapes
:
е
Egradients/loss/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapeNgradients/loss/softmax_cross_entropy_with_logits_grad/tuple/control_dependencyCgradients/loss/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
┐
(gradients/dense/dense_1/Softmax_grad/mulMulEgradients/loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshapedense/dense_1/Softmax*'
_output_shapes
:         *
T0
Е
:gradients/dense/dense_1/Softmax_grad/Sum/reduction_indicesConst*
valueB :
         *
dtype0*
_output_shapes
: 
ф
(gradients/dense/dense_1/Softmax_grad/SumSum(gradients/dense/dense_1/Softmax_grad/mul:gradients/dense/dense_1/Softmax_grad/Sum/reduction_indices*
T0*'
_output_shapes
:         *
	keep_dims(*

Tidx0
╥
(gradients/dense/dense_1/Softmax_grad/subSubEgradients/loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape(gradients/dense/dense_1/Softmax_grad/Sum*
T0*'
_output_shapes
:         
д
*gradients/dense/dense_1/Softmax_grad/mul_1Mul(gradients/dense/dense_1/Softmax_grad/subdense/dense_1/Softmax*
T0*'
_output_shapes
:         
з
0gradients/dense/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/dense/dense_1/Softmax_grad/mul_1*
T0*
data_formatNHWC*
_output_shapes
:
Э
5gradients/dense/dense_1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/dense/dense_1/BiasAdd_grad/BiasAddGrad+^gradients/dense/dense_1/Softmax_grad/mul_1
Ю
=gradients/dense/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity*gradients/dense/dense_1/Softmax_grad/mul_16^gradients/dense/dense_1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/dense/dense_1/Softmax_grad/mul_1*'
_output_shapes
:         
Я
?gradients/dense/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/dense/dense_1/BiasAdd_grad/BiasAddGrad6^gradients/dense/dense_1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/dense/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
р
*gradients/dense/dense_1/MatMul_grad/MatMulMatMul=gradients/dense/dense_1/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:         @
╓
,gradients/dense/dense_1/MatMul_grad/MatMul_1MatMuldense/dense/Tanh=gradients/dense/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:@*
transpose_b( *
T0
Ш
4gradients/dense/dense_1/MatMul_grad/tuple/group_depsNoOp+^gradients/dense/dense_1/MatMul_grad/MatMul-^gradients/dense/dense_1/MatMul_grad/MatMul_1
Ь
<gradients/dense/dense_1/MatMul_grad/tuple/control_dependencyIdentity*gradients/dense/dense_1/MatMul_grad/MatMul5^gradients/dense/dense_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/dense/dense_1/MatMul_grad/MatMul*'
_output_shapes
:         @
Щ
>gradients/dense/dense_1/MatMul_grad/tuple/control_dependency_1Identity,gradients/dense/dense_1/MatMul_grad/MatMul_15^gradients/dense/dense_1/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*?
_class5
31loc:@gradients/dense/dense_1/MatMul_grad/MatMul_1
╢
(gradients/dense/dense/Tanh_grad/TanhGradTanhGraddense/dense/Tanh<gradients/dense/dense_1/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:         @
г
.gradients/dense/dense/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients/dense/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Ч
3gradients/dense/dense/BiasAdd_grad/tuple/group_depsNoOp/^gradients/dense/dense/BiasAdd_grad/BiasAddGrad)^gradients/dense/dense/Tanh_grad/TanhGrad
Ц
;gradients/dense/dense/BiasAdd_grad/tuple/control_dependencyIdentity(gradients/dense/dense/Tanh_grad/TanhGrad4^gradients/dense/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/dense/dense/Tanh_grad/TanhGrad*'
_output_shapes
:         @
Ч
=gradients/dense/dense/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/dense/dense/BiasAdd_grad/BiasAddGrad4^gradients/dense/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*A
_class7
53loc:@gradients/dense/dense/BiasAdd_grad/BiasAddGrad
█
(gradients/dense/dense/MatMul_grad/MatMulMatMul;gradients/dense/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
T0*
transpose_a( *(
_output_shapes
:         А*
transpose_b(
╦
*gradients/dense/dense/MatMul_grad/MatMul_1MatMulconcat_2;gradients/dense/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	А@*
transpose_b( *
T0
Т
2gradients/dense/dense/MatMul_grad/tuple/group_depsNoOp)^gradients/dense/dense/MatMul_grad/MatMul+^gradients/dense/dense/MatMul_grad/MatMul_1
Х
:gradients/dense/dense/MatMul_grad/tuple/control_dependencyIdentity(gradients/dense/dense/MatMul_grad/MatMul3^gradients/dense/dense/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/dense/dense/MatMul_grad/MatMul*(
_output_shapes
:         А
Т
<gradients/dense/dense/MatMul_grad/tuple/control_dependency_1Identity*gradients/dense/dense/MatMul_grad/MatMul_13^gradients/dense/dense/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/dense/dense/MatMul_grad/MatMul_1*
_output_shapes
:	А@
^
gradients/concat_2_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_2_grad/modFloorModconcat_2/axisgradients/concat_2_grad/Rank*
T0*
_output_shapes
: 
e
gradients/concat_2_grad/ShapeShapeGatherNd*
_output_shapes
:*
T0*
out_type0
В
gradients/concat_2_grad/ShapeNShapeNGatherNd
GatherNd_1*
T0*
out_type0*
N* 
_output_shapes
::
╛
$gradients/concat_2_grad/ConcatOffsetConcatOffsetgradients/concat_2_grad/modgradients/concat_2_grad/ShapeN gradients/concat_2_grad/ShapeN:1*
N* 
_output_shapes
::
ш
gradients/concat_2_grad/SliceSlice:gradients/dense/dense/MatMul_grad/tuple/control_dependency$gradients/concat_2_grad/ConcatOffsetgradients/concat_2_grad/ShapeN*
Index0*
T0*(
_output_shapes
:         А
ю
gradients/concat_2_grad/Slice_1Slice:gradients/dense/dense/MatMul_grad/tuple/control_dependency&gradients/concat_2_grad/ConcatOffset:1 gradients/concat_2_grad/ShapeN:1*(
_output_shapes
:         А*
Index0*
T0
r
(gradients/concat_2_grad/tuple/group_depsNoOp^gradients/concat_2_grad/Slice ^gradients/concat_2_grad/Slice_1
ы
0gradients/concat_2_grad/tuple/control_dependencyIdentitygradients/concat_2_grad/Slice)^gradients/concat_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*0
_class&
$"loc:@gradients/concat_2_grad/Slice
ё
2gradients/concat_2_grad/tuple/control_dependency_1Identitygradients/concat_2_grad/Slice_1)^gradients/concat_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*2
_class(
&$loc:@gradients/concat_2_grad/Slice_1
И
gradients/GatherNd_grad/ShapeShape+blstm_1/bidirectional_rnn/fw/fw/transpose_1*
T0*
out_type0*
_output_shapes
:
╓
!gradients/GatherNd_grad/ScatterNd	ScatterNdstack0gradients/concat_2_grad/tuple/control_dependencygradients/GatherNd_grad/Shape*
T0*5
_output_shapes#
!:                  А*
Tindices0
v
gradients/GatherNd_1_grad/ShapeShapeblstm_1/ReverseSequence*
T0*
out_type0*
_output_shapes
:
▄
#gradients/GatherNd_1_grad/ScatterNd	ScatterNdstack2gradients/concat_2_grad/tuple/control_dependency_1gradients/GatherNd_1_grad/Shape*
Tindices0*
T0*5
_output_shapes#
!:                  А
░
Lgradients/blstm_1/bidirectional_rnn/fw/fw/transpose_1_grad/InvertPermutationInvertPermutation(blstm_1/bidirectional_rnn/fw/fw/concat_2*
T0*
_output_shapes
:
П
Dgradients/blstm_1/bidirectional_rnn/fw/fw/transpose_1_grad/transpose	Transpose!gradients/GatherNd_grad/ScatterNdLgradients/blstm_1/bidirectional_rnn/fw/fw/transpose_1_grad/InvertPermutation*
T0*5
_output_shapes#
!:                  А*
Tperm0
Ё
6gradients/blstm_1/ReverseSequence_grad/ReverseSequenceReverseSequence#gradients/GatherNd_1_grad/ScatterNdinputs/Placeholder_2*
seq_dim*

Tlen0*5
_output_shapes#
!:                  А*
	batch_dim *
T0
┌
ugradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3+blstm_1/bidirectional_rnn/fw/fw/TensorArray,blstm_1/bidirectional_rnn/fw/fw/while/Exit_2*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/fw/TensorArray*
source	gradients*
_output_shapes

:: 
Д
qgradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity,blstm_1/bidirectional_rnn/fw/fw/while/Exit_2v^gradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/fw/TensorArray*
_output_shapes
: 
Ь
{gradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3ugradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV36blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/rangeDgradients/blstm_1/bidirectional_rnn/fw/fw/transpose_1_grad/transposeqgradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
Д
gradients/zeros_like_1	ZerosLike,blstm_1/bidirectional_rnn/fw/fw/while/Exit_3*(
_output_shapes
:         А*
T0
Д
gradients/zeros_like_2	ZerosLike,blstm_1/bidirectional_rnn/fw/fw/while/Exit_4*(
_output_shapes
:         А*
T0
░
Lgradients/blstm_1/bidirectional_rnn/bw/bw/transpose_1_grad/InvertPermutationInvertPermutation(blstm_1/bidirectional_rnn/bw/bw/concat_2*
T0*
_output_shapes
:
д
Dgradients/blstm_1/bidirectional_rnn/bw/bw/transpose_1_grad/transpose	Transpose6gradients/blstm_1/ReverseSequence_grad/ReverseSequenceLgradients/blstm_1/bidirectional_rnn/bw/bw/transpose_1_grad/InvertPermutation*
T0*5
_output_shapes#
!:                  А*
Tperm0
ц
Bgradients/blstm_1/bidirectional_rnn/fw/fw/while/Exit_2_grad/b_exitEnter{gradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
У
Bgradients/blstm_1/bidirectional_rnn/fw/fw/while/Exit_3_grad/b_exitEntergradients/zeros_like_1*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*(
_output_shapes
:         А
У
Bgradients/blstm_1/bidirectional_rnn/fw/fw/while/Exit_4_grad/b_exitEntergradients/zeros_like_2*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*(
_output_shapes
:         А
┌
ugradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3+blstm_1/bidirectional_rnn/bw/bw/TensorArray,blstm_1/bidirectional_rnn/bw/bw/while/Exit_2*
_output_shapes

:: *>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/bw/TensorArray*
source	gradients
Д
qgradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity,blstm_1/bidirectional_rnn/bw/bw/while/Exit_2v^gradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
: 
Ь
{gradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3ugradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV36blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/rangeDgradients/blstm_1/bidirectional_rnn/bw/bw/transpose_1_grad/transposeqgradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
Д
gradients/zeros_like_3	ZerosLike,blstm_1/bidirectional_rnn/bw/bw/while/Exit_3*
T0*(
_output_shapes
:         А
Д
gradients/zeros_like_4	ZerosLike,blstm_1/bidirectional_rnn/bw/bw/while/Exit_4*(
_output_shapes
:         А*
T0
О
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switchMergeBgradients/blstm_1/bidirectional_rnn/fw/fw/while/Exit_2_grad/b_exitMgradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
а
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switchMergeBgradients/blstm_1/bidirectional_rnn/fw/fw/while/Exit_3_grad/b_exitMgradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_3_grad_1/NextIteration*
T0*
N**
_output_shapes
:         А: 
а
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switchMergeBgradients/blstm_1/bidirectional_rnn/fw/fw/while/Exit_4_grad/b_exitMgradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_4_grad_1/NextIteration*
T0*
N**
_output_shapes
:         А: 
ц
Bgradients/blstm_1/bidirectional_rnn/bw/bw/while/Exit_2_grad/b_exitEnter{gradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
У
Bgradients/blstm_1/bidirectional_rnn/bw/bw/while/Exit_3_grad/b_exitEntergradients/zeros_like_3*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*(
_output_shapes
:         А*
T0*
is_constant( 
У
Bgradients/blstm_1/bidirectional_rnn/bw/bw/while/Exit_4_grad/b_exitEntergradients/zeros_like_4*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*(
_output_shapes
:         А
и
Cgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/SwitchSwitchFgradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switchgradients/b_count_2*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: : 
Ы
Mgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_depsNoOpD^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch
Є
Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependencyIdentityCgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/SwitchN^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: 
Ў
Wgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1IdentityEgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch:1N^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: 
╠
Cgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3_grad/SwitchSwitchFgradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switchgradients/b_count_2*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*<
_output_shapes*
(:         А:         А
Ы
Mgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_depsNoOpD^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch
Д
Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependencyIdentityCgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3_grad/SwitchN^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*(
_output_shapes
:         А
И
Wgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1IdentityEgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch:1N^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*(
_output_shapes
:         А
╠
Cgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4_grad/SwitchSwitchFgradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switchgradients/b_count_2*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch*<
_output_shapes*
(:         А:         А
Ы
Mgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_depsNoOpD^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch
Д
Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependencyIdentityCgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4_grad/SwitchN^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch*(
_output_shapes
:         А
И
Wgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1IdentityEgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch:1N^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch
О
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switchMergeBgradients/blstm_1/bidirectional_rnn/bw/bw/while/Exit_2_grad/b_exitMgradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_2_grad_1/NextIteration*
N*
_output_shapes
: : *
T0
а
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switchMergeBgradients/blstm_1/bidirectional_rnn/bw/bw/while/Exit_3_grad/b_exitMgradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_3_grad_1/NextIteration*
N**
_output_shapes
:         А: *
T0
а
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switchMergeBgradients/blstm_1/bidirectional_rnn/bw/bw/while/Exit_4_grad/b_exitMgradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_4_grad_1/NextIteration*
T0*
N**
_output_shapes
:         А: 
┴
Agradients/blstm_1/bidirectional_rnn/fw/fw/while/Enter_2_grad/ExitExitUgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
╙
Agradients/blstm_1/bidirectional_rnn/fw/fw/while/Enter_3_grad/ExitExitUgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
╙
Agradients/blstm_1/bidirectional_rnn/fw/fw/while/Enter_4_grad/ExitExitUgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
и
Cgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/SwitchSwitchFgradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switchgradients/b_count_6*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: : 
Ы
Mgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_depsNoOpD^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch
Є
Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependencyIdentityCgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/SwitchN^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: 
Ў
Wgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1IdentityEgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch:1N^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: 
╠
Cgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3_grad/SwitchSwitchFgradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switchgradients/b_count_6*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*<
_output_shapes*
(:         А:         А
Ы
Mgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_depsNoOpD^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch
Д
Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependencyIdentityCgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3_grad/SwitchN^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*(
_output_shapes
:         А
И
Wgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1IdentityEgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch:1N^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*(
_output_shapes
:         А
╠
Cgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4_grad/SwitchSwitchFgradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switchgradients/b_count_6*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch*<
_output_shapes*
(:         А:         А
Ы
Mgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_depsNoOpD^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch
Д
Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependencyIdentityCgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4_grad/SwitchN^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch
И
Wgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1IdentityEgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch:1N^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch*(
_output_shapes
:         А
г
Аgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter+blstm_1/bidirectional_rnn/fw/fw/TensorArray*
is_constant(*M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
parallel_iterations 
ъ
zgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Аgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterWgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1*H
_class>
<:loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
source	gradients*
_output_shapes

:: 
├
vgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityWgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1{^gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
_output_shapes
: 
А
pgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*C
_class9
75loc:@blstm_1/bidirectional_rnn/fw/fw/while/Identity_1*
valueB :
         *
dtype0*
_output_shapes
: 
ё
pgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2pgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*C
_class9
75loc:@blstm_1/bidirectional_rnn/fw/fw/while/Identity_1*

stack_name *
_output_shapes
:
Г
pgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterpgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
э
vgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2pgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter0blstm_1/bidirectional_rnn/fw/fw/while/Identity_1^gradients/Add*
T0*
swap_memory( *
_output_shapes
: 
Ш
{gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterpgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╣
ugradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2{gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
: 
б
jgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3zgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3ugradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2vgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:         А
╕
igradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpX^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1k^gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
З
qgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityjgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3j^gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*}
_classs
qoloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*(
_output_shapes
:         А
└
sgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityWgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1j^gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: 
▐
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/ConstConst*C
_class9
75loc:@blstm_1/bidirectional_rnn/fw/fw/while/Identity_3*
valueB :
         *
dtype0*
_output_shapes
: 
н
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_accStackV2Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Const*
	elem_type0*C
_class9
75loc:@blstm_1/bidirectional_rnn/fw/fw/while/Identity_3*

stack_name *
_output_shapes
:
┐
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/EnterEnterNgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╗
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPushV2StackPushV2Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Enter0blstm_1/bidirectional_rnn/fw/fw/while/Identity_3^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:         А
╘
Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2/EnterEnterNgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
З
Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         А*
	elem_type0
▌
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like	ZerosLikeSgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2*(
_output_shapes
:         А*
T0
▄
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/ConstConst*E
_class;
97loc:@blstm_1/bidirectional_rnn/fw/fw/while/GreaterEqual*
valueB :
         *
dtype0*
_output_shapes
: 
з
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_accStackV2Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Const*
	elem_type0
*E
_class;
97loc:@blstm_1/bidirectional_rnn/fw/fw/while/GreaterEqual*

stack_name *
_output_shapes
:
╖
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/EnterEnterJgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
е
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2StackPushV2Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter2blstm_1/bidirectional_rnn/fw/fw/while/GreaterEqual^gradients/Add*
T0
*
swap_memory( *
_output_shapes
:
╠
Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2/EnterEnterJgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
я
Ogradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2
StackPopV2Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0

ї
Dgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectSelectOgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Wgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like*
T0*(
_output_shapes
:         А
ў
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1SelectOgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_likeWgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
ц
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_depsNoOpE^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectG^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
Е
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependencyIdentityDgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectO^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select*(
_output_shapes
:         А
Л
Xgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependency_1IdentityFgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1O^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1*(
_output_shapes
:         А
▐
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/ConstConst*C
_class9
75loc:@blstm_1/bidirectional_rnn/fw/fw/while/Identity_4*
valueB :
         *
dtype0*
_output_shapes
: 
н
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_accStackV2Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Const*

stack_name *
_output_shapes
:*
	elem_type0*C
_class9
75loc:@blstm_1/bidirectional_rnn/fw/fw/while/Identity_4
┐
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/EnterEnterNgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╗
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPushV2StackPushV2Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Enter0blstm_1/bidirectional_rnn/fw/fw/while/Identity_4^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:         А
╘
Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2/EnterEnterNgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
З
Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         А*
	elem_type0
▌
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like	ZerosLikeSgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:         А
ї
Dgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectSelectOgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Wgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like*(
_output_shapes
:         А*
T0
ў
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1SelectOgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_likeWgradients/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
ц
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_depsNoOpE^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectG^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1
Е
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependencyIdentityDgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectO^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/Select*(
_output_shapes
:         А
Л
Xgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency_1IdentityFgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1O^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1
┴
Agradients/blstm_1/bidirectional_rnn/bw/bw/while/Enter_2_grad/ExitExitUgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
╙
Agradients/blstm_1/bidirectional_rnn/bw/bw/while/Enter_3_grad/ExitExitUgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
╙
Agradients/blstm_1/bidirectional_rnn/bw/bw/while/Enter_4_grad/ExitExitUgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
м
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/EnterEnter%blstm_1/bidirectional_rnn/fw/fw/zeros*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*(
_output_shapes
:         А
ф
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like	ZerosLikeLgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/Enter^gradients/Sub*
T0*(
_output_shapes
:         А
Л
Bgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/SelectSelectOgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2qgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyFgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like*
T0*(
_output_shapes
:         А
Н
Dgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/Select_1SelectOgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/zeros_likeqgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
р
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_depsNoOpC^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/SelectE^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/Select_1
¤
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependencyIdentityBgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/SelectM^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/Select*(
_output_shapes
:         А
Г
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency_1IdentityDgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/Select_1M^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*W
_classM
KIloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/Select_1
г
Аgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter+blstm_1/bidirectional_rnn/bw/bw/TensorArray*
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context
ъ
zgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Аgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterWgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1*H
_class>
<:loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
source	gradients*
_output_shapes

:: 
├
vgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityWgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1{^gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*H
_class>
<:loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2
А
pgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*C
_class9
75loc:@blstm_1/bidirectional_rnn/bw/bw/while/Identity_1*
valueB :
         *
dtype0*
_output_shapes
: 
ё
pgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2pgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*C
_class9
75loc:@blstm_1/bidirectional_rnn/bw/bw/while/Identity_1*

stack_name *
_output_shapes
:*
	elem_type0
Г
pgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterpgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
я
vgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2pgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter0blstm_1/bidirectional_rnn/bw/bw/while/Identity_1^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
: 
Ш
{gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterpgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
╗
ugradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2{gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
: *
	elem_type0
б
jgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3zgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3ugradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2vgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:         А
╕
igradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpX^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1k^gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
З
qgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityjgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3j^gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*}
_classs
qoloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
└
sgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityWgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1j^gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: 
▐
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/ConstConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@blstm_1/bidirectional_rnn/bw/bw/while/Identity_3*
valueB :
         
н
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_accStackV2Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Const*

stack_name *
_output_shapes
:*
	elem_type0*C
_class9
75loc:@blstm_1/bidirectional_rnn/bw/bw/while/Identity_3
┐
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/EnterEnterNgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
╜
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPushV2StackPushV2Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Enter0blstm_1/bidirectional_rnn/bw/bw/while/Identity_3^gradients/Add_1*
swap_memory( *(
_output_shapes
:         А*
T0
╘
Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2/EnterEnterNgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Й
Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
▌
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like	ZerosLikeSgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:         А
▄
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/ConstConst*
dtype0*
_output_shapes
: *E
_class;
97loc:@blstm_1/bidirectional_rnn/bw/bw/while/GreaterEqual*
valueB :
         
з
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_accStackV2Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Const*E
_class;
97loc:@blstm_1/bidirectional_rnn/bw/bw/while/GreaterEqual*

stack_name *
_output_shapes
:*
	elem_type0

╖
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/EnterEnterJgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
з
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2StackPushV2Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter2blstm_1/bidirectional_rnn/bw/bw/while/GreaterEqual^gradients/Add_1*
swap_memory( *
_output_shapes
:*
T0

╠
Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2/EnterEnterJgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
ё
Ogradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2
StackPopV2Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
*
_output_shapes
:
ї
Dgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectSelectOgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Wgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like*
T0*(
_output_shapes
:         А
ў
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1SelectOgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_likeWgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
ц
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_depsNoOpE^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectG^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1
Е
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependencyIdentityDgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectO^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select*(
_output_shapes
:         А
Л
Xgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependency_1IdentityFgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1O^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1
▐
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/ConstConst*C
_class9
75loc:@blstm_1/bidirectional_rnn/bw/bw/while/Identity_4*
valueB :
         *
dtype0*
_output_shapes
: 
н
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_accStackV2Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Const*C
_class9
75loc:@blstm_1/bidirectional_rnn/bw/bw/while/Identity_4*

stack_name *
_output_shapes
:*
	elem_type0
┐
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/EnterEnterNgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
╜
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPushV2StackPushV2Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Enter0blstm_1/bidirectional_rnn/bw/bw/while/Identity_4^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:         А
╘
Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2/EnterEnterNgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Й
Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
▌
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like	ZerosLikeSgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:         А
ї
Dgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectSelectOgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Wgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like*
T0*(
_output_shapes
:         А
ў
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1SelectOgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_likeWgradients/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
ц
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_depsNoOpE^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectG^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1
Е
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependencyIdentityDgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectO^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/Select*(
_output_shapes
:         А
Л
Xgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency_1IdentityFgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1O^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1*(
_output_shapes
:         А
▀
gradients/AddNAddNXgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency_1Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency_1*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1*
N*(
_output_shapes
:         А
├
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ShapeShape9blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
┬
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1Shape6blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*
_output_shapes
:*
T0*
out_type0
К
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape*
valueB :
         
ы
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape
у
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnter`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
ы
fgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterJgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape^gradients/Add*
T0*
swap_memory( *
_output_shapes
:
°
kgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnter`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Э
egradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2kgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
О
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1*
valueB :
         
ё
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
ч
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enterbgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
ё
hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1^gradients/Add*
swap_memory( *
_output_shapes
:*
T0
№
mgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterbgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
б
ggradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2mgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
А
Zgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsegradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2ggradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
ф
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/ConstConst*I
_class?
=;loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*
valueB :
         *
dtype0*
_output_shapes
: 
│
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_accStackV2Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*I
_class?
=;loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1
┐
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/EnterEnterNgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
┴
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Enter6blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1^gradients/Add*
swap_memory( *(
_output_shapes
:         А*
T0
╘
Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterNgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
З
Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:         А
ч
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/MulMulgradients/AddNSgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*(
_output_shapes
:         А
╡
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/SumSumHgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/MulZgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╔
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ReshapeReshapeHgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Sumegradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
щ
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*L
_classB
@>loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2*
valueB :
         *
dtype0*
_output_shapes
: 
║
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Const*L
_classB
@>loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0
├
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterPgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╚
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Enter9blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:         А
╪
[gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterPgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Л
Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2[gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         А*
	elem_type0
ы
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1MulUgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2gradients/AddN*(
_output_shapes
:         А*
T0
╗
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Sum_1SumJgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1\gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╧
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1ReshapeJgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Sum_1ggradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
¤
Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/group_depsNoOpM^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ReshapeO^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1
г
]gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityLgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ReshapeV^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape*(
_output_shapes
:         А
й
_gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityNgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1V^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*a
_classW
USloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1
Ї
Mgradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_2_grad_1/NextIterationNextIterationsgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
м
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/EnterEnter%blstm_1/bidirectional_rnn/bw/bw/zeros*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*(
_output_shapes
:         А*
T0*
is_constant(
ц
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like	ZerosLikeLgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/Enter^gradients/Sub_1*
T0*(
_output_shapes
:         А
Л
Bgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/SelectSelectOgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2qgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyFgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like*
T0*(
_output_shapes
:         А
Н
Dgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/Select_1SelectOgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/zeros_likeqgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
р
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_depsNoOpC^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/SelectE^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/Select_1
¤
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependencyIdentityBgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/SelectM^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/Select*(
_output_shapes
:         А
Г
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency_1IdentityDgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/Select_1M^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/Select_1*(
_output_shapes
:         А
╠
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradUgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2]gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
├
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradSgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2_gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
с
gradients/AddN_1AddNXgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency_1Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency_1*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1*
N*(
_output_shapes
:         А
├
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ShapeShape9blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
┬
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1Shape6blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*
T0*
out_type0*
_output_shapes
:
К
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
ы
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*
	elem_type0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape*

stack_name *
_output_shapes
:
у
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnter`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
э
fgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterJgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
°
kgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnter`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Я
egradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2kgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
О
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1*
valueB :
         
ё
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
ч
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enterbgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
є
hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
№
mgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterbgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
г
ggradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2mgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
А
Zgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsegradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2ggradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
ф
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/ConstConst*I
_class?
=;loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*
valueB :
         *
dtype0*
_output_shapes
: 
│
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_accStackV2Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Const*I
_class?
=;loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
┐
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/EnterEnterNgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
├
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Enter6blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1^gradients/Add_1*
swap_memory( *(
_output_shapes
:         А*
T0
╘
Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterNgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Й
Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
щ
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/MulMulgradients/AddN_1Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2*(
_output_shapes
:         А*
T0
╡
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/SumSumHgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/MulZgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╔
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ReshapeReshapeHgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Sumegradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
щ
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*L
_classB
@>loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2*
valueB :
         *
dtype0*
_output_shapes
: 
║
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*L
_classB
@>loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2
├
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterPgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
╩
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Enter9blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:         А
╪
[gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterPgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Н
Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2[gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
э
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1MulUgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2gradients/AddN_1*
T0*(
_output_shapes
:         А
╗
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Sum_1SumJgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1\gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╧
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1ReshapeJgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Sum_1ggradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
¤
Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/group_depsNoOpM^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ReshapeO^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1
г
]gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityLgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ReshapeV^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape
й
_gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityNgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1V^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*a
_classW
USloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1
Ї
Mgradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_2_grad_1/NextIterationNextIterationsgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
┘
gradients/AddN_2AddNXgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependency_1Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1_grad/TanhGrad*
N*(
_output_shapes
:         А*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
╜
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ShapeShape3blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul*
_output_shapes
:*
T0*
out_type0
┴
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1Shape5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1*
_output_shapes
:*
T0*
out_type0
К
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
ы
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape
у
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnter`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
ы
fgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterJgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape^gradients/Add*
swap_memory( *
_output_shapes
:*
T0
°
kgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Э
egradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2kgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
О
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
ё
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*
_output_shapes
:*
	elem_type0*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1*

stack_name 
ч
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enterbgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
ё
hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1^gradients/Add*
swap_memory( *
_output_shapes
:*
T0
№
mgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterbgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
б
ggradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2mgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
А
Zgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsegradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2ggradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
¤
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/SumSumgradients/AddN_2Zgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╔
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ReshapeReshapeHgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Sumegradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:         А*
T0*
Tshape0
Б
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_2\gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╧
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1ReshapeJgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Sum_1ggradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*(
_output_shapes
:         А*
T0*
Tshape0
¤
Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/group_depsNoOpM^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ReshapeO^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1
г
]gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependencyIdentityLgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ReshapeV^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/group_deps*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape*(
_output_shapes
:         А*
T0
й
_gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependency_1IdentityNgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1V^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*a
_classW
USloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1
╠
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradUgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2]gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
├
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradSgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2_gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
┐
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ShapeShape7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
║
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1Shape0blstm_1/bidirectional_rnn/fw/fw/while/Identity_3*
T0*
out_type0*
_output_shapes
:
Ж
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
х
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
▀
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnter^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(*
parallel_iterations 
х
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterHgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape^gradients/Add*
swap_memory( *
_output_shapes
:*
T0
Ї
igradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Щ
cgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
К
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
ы
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
у
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enter`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
ы
fgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1^gradients/Add*
swap_memory( *
_output_shapes
:*
T0
°
kgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Э
egradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2kgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
·
Xgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2egradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
┤
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/MulMul]gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependencySgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2*(
_output_shapes
:         А*
T0
п
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/SumSumFgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/MulXgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
├
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ReshapeReshapeFgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Sumcgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
х
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/ConstConst*J
_class@
><loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid*
valueB :
         *
dtype0*
_output_shapes
: 
┤
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_accStackV2Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Const*J
_class@
><loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
┐
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/EnterEnterNgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0
┬
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Enter7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid^gradients/Add*
swap_memory( *(
_output_shapes
:         А*
T0
╘
Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterNgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
З
Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         А*
	elem_type0
╢
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1MulSgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2]gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
╡
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Sum_1SumHgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1Zgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╔
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1ReshapeHgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Sum_1egradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*(
_output_shapes
:         А*
T0*
Tshape0
ў
Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/group_depsNoOpK^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ReshapeM^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1
Ы
[gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/control_dependencyIdentityJgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ReshapeT^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape*(
_output_shapes
:         А
б
]gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/control_dependency_1IdentityLgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1T^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/group_deps*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1*(
_output_shapes
:         А*
T0
├
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ShapeShape9blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
└
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1Shape4blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
К
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
ы
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
у
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnter`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
ы
fgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterJgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape^gradients/Add*
T0*
swap_memory( *
_output_shapes
:
°
kgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Э
egradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2kgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
О
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
ё
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
ч
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enterbgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
ё
hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1^gradients/Add*
T0*
swap_memory( *
_output_shapes
:
№
mgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterbgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
б
ggradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2mgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
А
Zgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsegradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2ggradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
т
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/ConstConst*G
_class=
;9loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh*
valueB :
         *
dtype0*
_output_shapes
: 
▒
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_accStackV2Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*G
_class=
;9loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh
┐
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/EnterEnterNgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
┐
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Enter4blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:         А
╘
Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterNgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
З
Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         А*
	elem_type0
╕
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/MulMul_gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependency_1Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2*(
_output_shapes
:         А*
T0
╡
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/SumSumHgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/MulZgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╔
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ReshapeReshapeHgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Sumegradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
щ
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *L
_classB
@>loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1*
valueB :
         
║
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Const*L
_classB
@>loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
├
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterPgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╚
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Enter9blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:         А
╪
[gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterPgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Л
Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2[gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:         А
╝
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1MulUgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2_gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
╗
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Sum_1SumJgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1\gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╧
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1ReshapeJgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Sum_1ggradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
¤
Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/group_depsNoOpM^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ReshapeO^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1
г
]gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityLgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ReshapeV^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape*(
_output_shapes
:         А
й
_gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityNgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1V^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*a
_classW
USloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1
┘
gradients/AddN_3AddNXgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependency_1Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1_grad/TanhGrad*
T0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1*
N*(
_output_shapes
:         А
╜
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ShapeShape3blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:
┴
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1Shape5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
К
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape*
valueB :
         
ы
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
у
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnter`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
э
fgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterJgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
°
kgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Я
egradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2kgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
О
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1*
valueB :
         
ё
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1
ч
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enterbgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
є
hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
№
mgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterbgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
г
ggradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2mgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
А
Zgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsegradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2ggradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
¤
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/SumSumgradients/AddN_3Zgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╔
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ReshapeReshapeHgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Sumegradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
Б
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_3\gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╧
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1ReshapeJgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Sum_1ggradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
¤
Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/group_depsNoOpM^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ReshapeO^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1
г
]gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependencyIdentityLgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ReshapeV^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape*(
_output_shapes
:         А
й
_gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependency_1IdentityNgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1V^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1*(
_output_shapes
:         А
╞
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradSgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2[gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ф
gradients/AddN_4AddNVgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependency]gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select*
N*(
_output_shapes
:         А
╠
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradUgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2]gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
┴
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_grad/TanhGradTanhGradSgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2_gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
┐
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ShapeShape7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
║
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1Shape0blstm_1/bidirectional_rnn/bw/bw/while/Identity_3*
T0*
out_type0*
_output_shapes
:
Ж
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
х
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape
▀
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnter^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
ч
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterHgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
Ї
igradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Ы
cgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
К
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
ы
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:
у
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enter`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
э
fgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
°
kgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Я
egradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2kgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
·
Xgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2egradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
┤
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/MulMul]gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependencySgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:         А
п
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/SumSumFgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/MulXgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
├
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ReshapeReshapeFgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Sumcgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
х
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/ConstConst*J
_class@
><loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid*
valueB :
         *
dtype0*
_output_shapes
: 
┤
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_accStackV2Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Const*J
_class@
><loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
┐
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/EnterEnterNgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
─
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Enter7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid^gradients/Add_1*
swap_memory( *(
_output_shapes
:         А*
T0
╘
Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterNgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Й
Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
╢
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1MulSgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2]gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
╡
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Sum_1SumHgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1Zgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╔
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1ReshapeHgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Sum_1egradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
ў
Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/group_depsNoOpK^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ReshapeM^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1
Ы
[gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/control_dependencyIdentityJgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ReshapeT^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape
б
]gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/control_dependency_1IdentityLgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1T^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1*(
_output_shapes
:         А
├
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ShapeShape9blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
└
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1Shape4blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
К
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
ы
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape
у
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnter`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
э
fgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterJgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
°
kgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Я
egradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2kgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
О
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
ё
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
ч
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enterbgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
є
hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1^gradients/Add_1*
swap_memory( *
_output_shapes
:*
T0
№
mgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterbgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
г
ggradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2mgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
А
Zgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsegradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2ggradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
т
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/ConstConst*G
_class=
;9loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*
valueB :
         *
dtype0*
_output_shapes
: 
▒
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_accStackV2Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*G
_class=
;9loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh
┐
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/EnterEnterNgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
┴
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Enter4blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh^gradients/Add_1*
swap_memory( *(
_output_shapes
:         А*
T0
╘
Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterNgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Й
Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
╕
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/MulMul_gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependency_1Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*(
_output_shapes
:         А
╡
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/SumSumHgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/MulZgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╔
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ReshapeReshapeHgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Sumegradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:         А*
T0*
Tshape0
щ
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *L
_classB
@>loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1*
valueB :
         
║
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Const*L
_classB
@>loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
├
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterPgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
╩
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Enter9blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:         А
╪
[gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterPgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Н
Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2[gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
╝
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1MulUgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2_gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
╗
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Sum_1SumJgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1\gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╧
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1ReshapeJgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Sum_1ggradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*(
_output_shapes
:         А*
T0*
Tshape0
¤
Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/group_depsNoOpM^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ReshapeO^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1
г
]gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityLgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ReshapeV^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape*(
_output_shapes
:         А
й
_gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityNgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1V^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1*(
_output_shapes
:         А
┐
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ShapeShape7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:
Э
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
Ж
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
х
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
▀
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnter^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
х
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterHgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape^gradients/Add*
T0*
swap_memory( *
_output_shapes
:
Ї
igradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Щ
cgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
▀
Xgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╗
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/SumSumRgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_grad/SigmoidGradXgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
├
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ReshapeReshapeFgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Sumcgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
┐
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Sum_1SumRgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_grad/SigmoidGradZgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ь
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1ReshapeHgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Sum_1Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ў
Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/group_depsNoOpK^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ReshapeM^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1
Ы
[gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/control_dependencyIdentityJgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ReshapeT^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape
П
]gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/control_dependency_1IdentityLgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1T^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1*
_output_shapes
: 
г
Mgradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_4*
T0*(
_output_shapes
:         А
╞
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradSgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2[gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
ф
gradients/AddN_5AddNVgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependency]gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select*
N*(
_output_shapes
:         А
╠
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradUgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2]gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
┴
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_grad/TanhGradTanhGradSgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2_gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
г
Qgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
╔
Kgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concatConcatV2Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradLgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_grad/TanhGrad[gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/control_dependencyTgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradQgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat/Const*
N*(
_output_shapes
:         А*

Tidx0*
T0
┐
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ShapeShape7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:
Я
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape_1Const^gradients/Sub_1*
valueB *
dtype0*
_output_shapes
: 
Ж
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
х
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape
▀
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnter^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
ч
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterHgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
Ї
igradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Ы
cgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
▀
Xgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╗
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/SumSumRgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_grad/SigmoidGradXgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
├
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ReshapeReshapeFgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Sumcgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:         А*
T0*
Tshape0
┐
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Sum_1SumRgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_grad/SigmoidGradZgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ь
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1ReshapeHgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Sum_1Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ў
Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/group_depsNoOpK^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ReshapeM^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1
Ы
[gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/control_dependencyIdentityJgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ReshapeT^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape*(
_output_shapes
:         А
П
]gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/control_dependency_1IdentityLgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1T^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1*
_output_shapes
: 
г
Mgradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_5*
T0*(
_output_shapes
:         А
ы
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradKgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:А
В
Wgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpS^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGradL^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat
е
_gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityKgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concatX^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat*(
_output_shapes
:         А
и
agradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityRgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGradX^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*e
_class[
YWloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGrad
е
Qgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat/ConstConst^gradients/Sub_1*
dtype0*
_output_shapes
: *
value	B :
╔
Kgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concatConcatV2Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradLgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_grad/TanhGrad[gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/control_dependencyTgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradQgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat/Const*
N*(
_output_shapes
:         А*

Tidx0*
T0
╖
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul/EnterEnter2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context* 
_output_shapes
:
АА
ф
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMulMatMul_gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyRgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul/Enter*
T0*
transpose_a( *(
_output_shapes
:         А*
transpose_b(
ъ
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*I
_class?
=;loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat*
valueB :
         *
dtype0*
_output_shapes
: 
┐
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Const*I
_class?
=;loc:@blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
╦
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterTgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
═
Zgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Enter6blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:         А
р
_gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterTgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
У
Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2_gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         А*
	elem_type0
х
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1MatMulYgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2_gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
АА*
transpose_b( 
■
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/group_depsNoOpM^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMulO^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1
е
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityLgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMulW^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul*(
_output_shapes
:         А
г
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityNgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1W^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
АА*
T0*a
_classW
USloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1
б
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
_output_shapes	
:А*
valueBА*    
╘
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterRgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes	
:А*
T0*
is_constant( 
└
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeTgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1Zgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:А: 
ї
Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchTgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
:А:А
╖
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/AddAddUgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/Switch:1agradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:А
у
Zgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationPgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:А*
T0
╫
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitSgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:А
ы
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradKgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:А*
T0
В
Wgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpS^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGradL^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat
е
_gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityKgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concatX^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*^
_classT
RPloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat
и
agradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityRgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGradX^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
Э
Kgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConstConst^gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
Ь
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
П
Igradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/modFloorModKgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConstJgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
╝
Kgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeShape1blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul*
_output_shapes
:*
T0*
out_type0
у
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/ConstConst*D
_class:
86loc:@blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul*
valueB :
         *
dtype0*
_output_shapes
: 
╢
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_accStackV2Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Const*D
_class:
86loc:@blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul*

stack_name *
_output_shapes
:*
	elem_type0
╟
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/EnterEnterRgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
─
Xgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Enter1blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:         А
▄
]gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterRgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
П
Wgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2]gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         А*
	elem_type0
╚
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeNShapeNWgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2*
T0*
out_type0*
N* 
_output_shapes
::
Ў
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConcatOffsetConcatOffsetIgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/modLgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeNNgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
Ц
Kgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/SliceSlice^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependencyRgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConcatOffsetLgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN*
Index0*
T0*(
_output_shapes
:         А
Ь
Mgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1Slice^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependencyTgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConcatOffset:1Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:         А
№
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/group_depsNoOpL^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/SliceN^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1
г
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/control_dependencyIdentityKgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/SliceW^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice*(
_output_shapes
:         А
й
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/control_dependency_1IdentityMgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1W^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1*(
_output_shapes
:         А
к
Qgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_accConst*
dtype0* 
_output_shapes
:
АА*
valueB
АА*    
╫
Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterQgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context* 
_output_shapes
:
АА
┬
Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeSgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_1Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/NextIteration*
N*"
_output_shapes
:
АА: *
T0
¤
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchSgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
АА:
АА
╣
Ogradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/AddAddTgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/Switch:1`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
АА
ц
Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationOgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
АА
┌
Sgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitRgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
АА
╖
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul/EnterEnter2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/read*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context* 
_output_shapes
:
АА*
T0*
is_constant(
ф
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMulMatMul_gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyRgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:         А
ъ
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*I
_class?
=;loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat*
valueB :
         *
dtype0*
_output_shapes
: 
┐
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*I
_class?
=;loc:@blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat
╦
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterTgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
╧
Zgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Enter6blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat^gradients/Add_1*
swap_memory( *(
_output_shapes
:         А*
T0
р
_gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterTgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Х
Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2_gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
х
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1MatMulYgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2_gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_a(* 
_output_shapes
:
АА*
transpose_b( *
T0
■
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/group_depsNoOpM^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMulO^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1
е
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityLgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMulW^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul*(
_output_shapes
:         А
г
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityNgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1W^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
б
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueBА*    *
dtype0*
_output_shapes	
:А
╘
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterRgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes	
:А
└
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeTgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1Zgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes
	:А: *
T0
ї
Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchTgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_6*"
_output_shapes
:А:А*
T0
╖
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/AddAddUgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/Switch:1agradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:А*
T0
у
Zgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationPgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:А
╫
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitSgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:А
╖
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ShapeShape1blstm_1/bidirectional_rnn/fw/fw/while/dropout/div*
_output_shapes
:*
T0*
out_type0
╗
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1Shape3blstm_1/bidirectional_rnn/fw/fw/while/dropout/Floor*
T0*
out_type0*
_output_shapes
:
В
\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/ConstConst*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
▀
\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_accStackV2\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Const*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
█
\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/EnterEnter\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
▀
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/EnterFgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape^gradients/Add*
T0*
swap_memory( *
_output_shapes
:
Ё
ggradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Х
agradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
Ж
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Const_1Const*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
х
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1StackV2^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1
▀
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1Enter^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
х
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1^gradients/Add*
T0*
swap_memory( *
_output_shapes
:
Ї
igradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Щ
cgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2igradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
Ї
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2cgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
▌
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/ConstConst*F
_class<
:8loc:@blstm_1/bidirectional_rnn/fw/fw/while/dropout/Floor*
valueB :
         *
dtype0*
_output_shapes
: 
и
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_accStackV2Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Const*
	elem_type0*F
_class<
:8loc:@blstm_1/bidirectional_rnn/fw/fw/while/dropout/Floor*

stack_name *
_output_shapes
:
╖
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/EnterEnterJgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╢
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPushV2StackPushV2Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Enter3blstm_1/bidirectional_rnn/fw/fw/while/dropout/Floor^gradients/Add*
swap_memory( *(
_output_shapes
:         А*
T0
╠
Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2/EnterEnterJgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
 
Ogradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2
StackPopV2Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         А*
	elem_type0
п
Dgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/MulMul^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/control_dependencyOgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2*(
_output_shapes
:         А*
T0
й
Dgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/SumSumDgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/MulVgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╜
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ReshapeReshapeDgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Sumagradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
▌
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@blstm_1/bidirectional_rnn/fw/fw/while/dropout/div*
valueB :
         
к
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_accStackV2Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*D
_class:
86loc:@blstm_1/bidirectional_rnn/fw/fw/while/dropout/div
╗
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/EnterEnterLgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╕
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPushV2StackPushV2Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Enter1blstm_1/bidirectional_rnn/fw/fw/while/dropout/div^gradients/Add*
swap_memory( *(
_output_shapes
:         А*
T0
╨
Wgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2/EnterEnterLgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Г
Qgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2
StackPopV2Wgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         А*
	elem_type0
│
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1MulQgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
п
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Sum_1SumFgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1Xgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
├
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1ReshapeFgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Sum_1cgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
ё
Qgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/group_depsNoOpI^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ReshapeK^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1
У
Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/control_dependencyIdentityHgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ReshapeR^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape*(
_output_shapes
:         А
Щ
[gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/control_dependency_1IdentityJgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1R^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1*(
_output_shapes
:         А
ч
gradients/AddN_6AddNVgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/Select*
N*(
_output_shapes
:         А
Я
Kgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConstConst^gradients/Sub_1*
value	B :*
dtype0*
_output_shapes
: 
Ю
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/RankConst^gradients/Sub_1*
value	B :*
dtype0*
_output_shapes
: 
П
Igradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/modFloorModKgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConstJgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
╝
Kgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeShape1blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul*
T0*
out_type0*
_output_shapes
:
у
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/ConstConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul*
valueB :
         
╢
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_accStackV2Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Const*

stack_name *
_output_shapes
:*
	elem_type0*D
_class:
86loc:@blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul
╟
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/EnterEnterRgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
╞
Xgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Enter1blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:         А
▄
]gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterRgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
С
Wgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2]gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
╚
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeNShapeNWgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2*
T0*
out_type0*
N* 
_output_shapes
::
Ў
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConcatOffsetConcatOffsetIgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/modLgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeNNgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
Ц
Kgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/SliceSlice^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependencyRgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConcatOffsetLgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN*(
_output_shapes
:         А*
Index0*
T0
Ь
Mgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1Slice^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependencyTgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConcatOffset:1Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:         А
№
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/group_depsNoOpL^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/SliceN^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1
г
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/control_dependencyIdentityKgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/SliceW^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice*(
_output_shapes
:         А
й
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/control_dependency_1IdentityMgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1W^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1*(
_output_shapes
:         А
к
Qgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
АА*    *
dtype0* 
_output_shapes
:
АА
╫
Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterQgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context* 
_output_shapes
:
АА*
T0*
is_constant( 
┬
Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeSgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_1Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
АА: 
¤
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchSgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_6*
T0*,
_output_shapes
:
АА:
АА
╣
Ogradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/AddAddTgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/Switch:1`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
АА*
T0
ц
Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationOgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
АА
┌
Sgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitRgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
АА
╜
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/ShapeShape7blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3*
_output_shapes
:*
T0*
out_type0
Ы
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
В
\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape*
valueB :
         
▀
\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_accStackV2\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/Const*
	elem_type0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape*

stack_name *
_output_shapes
:
█
\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/EnterEnter\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
▀
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/EnterFgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape^gradients/Add*
swap_memory( *
_output_shapes
:*
T0
Ё
ggradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Х
agradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
┘
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
г
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv/ConstConst^gradients/Sub*
dtype0*
_output_shapes
: *
valueB
 *fff?
▒
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDivRealDivYgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/control_dependencyNgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv/Const*
T0*(
_output_shapes
:         А
н
Dgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/SumSumHgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDivVgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╜
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/ReshapeReshapeDgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Sumagradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
с
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/ConstConst*J
_class@
><loc:@blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3*
valueB :
         *
dtype0*
_output_shapes
: 
м
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/f_accStackV2Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/Const*J
_class@
><loc:@blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0
╖
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/EnterEnterJgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/f_acc*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
║
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPushV2StackPushV2Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/Enter7blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:         А
а
gradients/NextIterationNextIterationgradients/AddQ^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2U^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPushV2U^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPushV2w^gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2c^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2Q^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPushV2c^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2e^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1Q^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPushV2S^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPushV2[^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2g^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2i^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1e^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2Y^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPushV2g^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2i^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1U^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPushV2W^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2g^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2i^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1U^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPushV2W^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2e^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2g^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1U^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPushV2*
T0*
_output_shapes
: 
╠
Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPopV2/EnterEnterJgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
 
Ogradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPopV2
StackPopV2Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub*(
_output_shapes
:         А*
	elem_type0
▒
qgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerP^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2T^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2T^gradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2v^gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2b^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2P^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPopV2b^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2d^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1P^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2R^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2Z^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2f^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2h^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1d^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2X^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2f^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2h^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1T^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2V^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2f^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2h^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1T^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2V^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2d^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2f^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1T^gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
╬
gradients/NextIteration_1NextIterationgradients/Subr^gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
╧
Dgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/NegNegOgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPopV2*
T0*(
_output_shapes
:         А
Ю
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv_1RealDivDgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/NegNgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv/Const*
T0*(
_output_shapes
:         А
д
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv_2RealDivJgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv_1Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv/Const*
T0*(
_output_shapes
:         А
е
Dgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/mulMulYgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/control_dependencyJgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv_2*
T0*(
_output_shapes
:         А
н
Fgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Sum_1SumDgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/mulXgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ц
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Reshape_1ReshapeFgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Sum_1Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ё
Qgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/tuple/group_depsNoOpI^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/ReshapeK^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Reshape_1
У
Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/tuple/control_dependencyIdentityHgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/ReshapeR^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Reshape*(
_output_shapes
:         А
З
[gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/tuple/control_dependency_1IdentityJgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Reshape_1R^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/tuple/group_deps*
_output_shapes
: *
T0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Reshape_1
г
Mgradients/blstm_1/bidirectional_rnn/fw/fw/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_6*(
_output_shapes
:         А*
T0
╖
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ShapeShape1blstm_1/bidirectional_rnn/bw/bw/while/dropout/div*
T0*
out_type0*
_output_shapes
:
╗
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1Shape3blstm_1/bidirectional_rnn/bw/bw/while/dropout/Floor*
T0*
out_type0*
_output_shapes
:
В
\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/ConstConst*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
▀
\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_accStackV2\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Const*
	elem_type0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape*

stack_name *
_output_shapes
:
█
\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/EnterEnter\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
с
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/EnterFgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
Ё
ggradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Ч
agradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
Ж
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Const_1Const*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
х
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1StackV2^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1*

stack_name *
_output_shapes
:
▀
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1Enter^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
ч
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
Ї
igradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Ы
cgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2igradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
Ї
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2cgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
▌
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/ConstConst*F
_class<
:8loc:@blstm_1/bidirectional_rnn/bw/bw/while/dropout/Floor*
valueB :
         *
dtype0*
_output_shapes
: 
и
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_accStackV2Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Const*
	elem_type0*F
_class<
:8loc:@blstm_1/bidirectional_rnn/bw/bw/while/dropout/Floor*

stack_name *
_output_shapes
:
╖
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/EnterEnterJgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
╕
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPushV2StackPushV2Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Enter3blstm_1/bidirectional_rnn/bw/bw/while/dropout/Floor^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:         А
╠
Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2/EnterEnterJgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Б
Ogradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2
StackPopV2Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
п
Dgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/MulMul^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/control_dependencyOgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2*
T0*(
_output_shapes
:         А
й
Dgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/SumSumDgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/MulVgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╜
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ReshapeReshapeDgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Sumagradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
▌
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@blstm_1/bidirectional_rnn/bw/bw/while/dropout/div*
valueB :
         
к
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_accStackV2Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Const*D
_class:
86loc:@blstm_1/bidirectional_rnn/bw/bw/while/dropout/div*

stack_name *
_output_shapes
:*
	elem_type0
╗
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/EnterEnterLgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
║
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPushV2StackPushV2Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Enter1blstm_1/bidirectional_rnn/bw/bw/while/dropout/div^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:         А
╨
Wgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2/EnterEnterLgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Е
Qgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2
StackPopV2Wgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
│
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1MulQgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
п
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Sum_1SumFgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1Xgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
├
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1ReshapeFgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Sum_1cgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
ё
Qgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/group_depsNoOpI^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ReshapeK^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1
У
Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/control_dependencyIdentityHgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ReshapeR^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape*(
_output_shapes
:         А
Щ
[gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/control_dependency_1IdentityJgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1R^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1*(
_output_shapes
:         А
ч
gradients/AddN_7AddNVgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/Select*
N*(
_output_shapes
:         А
Ъ
ngradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter-blstm_1/bidirectional_rnn/fw/fw/TensorArray_1*
T0*P
_classF
DBloc:@blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
:*M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context
┼
pgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterZblstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: *
T0*P
_classF
DBloc:@blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
parallel_iterations 
Ў
hgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3ngradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterpgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*P
_classF
DBloc:@blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
source	gradients*
_output_shapes

:: 
└
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentitypgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1i^gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*P
_classF
DBloc:@blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
_output_shapes
: 
├
jgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3hgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3ugradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Ygradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/tuple/control_dependencydgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
╜
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/ShapeShape7blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
Э
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape_1Const^gradients/Sub_1*
valueB *
dtype0*
_output_shapes
: 
В
\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape*
valueB :
         
▀
\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_accStackV2\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*Y
_classO
MKloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape
█
\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/EnterEnter\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
с
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/EnterFgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
Ё
ggradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Ч
agradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
┘
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
е
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv/ConstConst^gradients/Sub_1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
▒
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDivRealDivYgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/control_dependencyNgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv/Const*
T0*(
_output_shapes
:         А
н
Dgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/SumSumHgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDivVgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╜
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/ReshapeReshapeDgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Sumagradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
с
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/ConstConst*J
_class@
><loc:@blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3*
valueB :
         *
dtype0*
_output_shapes
: 
м
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/f_accStackV2Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/Const*

stack_name *
_output_shapes
:*
	elem_type0*J
_class@
><loc:@blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3
╖
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/EnterEnterJgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
╝
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPushV2StackPushV2Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/Enter7blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:         А
д
gradients/NextIteration_2NextIterationgradients/Add_1Q^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2U^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPushV2U^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPushV2w^gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2c^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2Q^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPushV2c^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2e^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1Q^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPushV2S^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPushV2[^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2g^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2i^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1e^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2Y^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPushV2g^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2i^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1U^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPushV2W^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2g^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2i^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1U^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPushV2W^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2e^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2g^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1U^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPushV2*
T0*
_output_shapes
: 
╠
Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPopV2/EnterEnterJgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Б
Ogradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPopV2
StackPopV2Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
▒
qgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerP^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2T^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2T^gradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2v^gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2b^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2P^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPopV2b^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2d^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1P^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2R^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2Z^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2f^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2h^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1d^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2X^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2f^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2h^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1T^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2V^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2f^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2h^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1T^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2V^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2d^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2f^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1T^gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
╨
gradients/NextIteration_3NextIterationgradients/Sub_1r^gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
╧
Dgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/NegNegOgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPopV2*
T0*(
_output_shapes
:         А
Ю
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv_1RealDivDgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/NegNgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv/Const*
T0*(
_output_shapes
:         А
д
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv_2RealDivJgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv_1Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv/Const*
T0*(
_output_shapes
:         А
е
Dgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/mulMulYgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/control_dependencyJgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv_2*
T0*(
_output_shapes
:         А
н
Fgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Sum_1SumDgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/mulXgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ц
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Reshape_1ReshapeFgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Sum_1Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ё
Qgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/tuple/group_depsNoOpI^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/ReshapeK^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Reshape_1
У
Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/tuple/control_dependencyIdentityHgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/ReshapeR^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*[
_classQ
OMloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Reshape
З
[gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/tuple/control_dependency_1IdentityJgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Reshape_1R^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Reshape_1*
_output_shapes
: 
г
Mgradients/blstm_1/bidirectional_rnn/bw/bw/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_7*(
_output_shapes
:         А*
T0
Щ
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╙
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterTgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
┴
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeVgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1\gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
я
Ugradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchVgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
T0*
_output_shapes
: : 
┐
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/AddAddWgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Switch:1jgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
т
\gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationRgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
╓
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitUgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
Ъ
ngradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter-blstm_1/bidirectional_rnn/bw/bw/TensorArray_1*
is_constant(*M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*P
_classF
DBloc:@blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
parallel_iterations 
┼
pgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterZblstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0*P
_classF
DBloc:@blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
parallel_iterations 
°
hgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3ngradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterpgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub_1*P
_classF
DBloc:@blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
source	gradients*
_output_shapes

:: 
└
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentitypgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1i^gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*P
_classF
DBloc:@blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
_output_shapes
: 
├
jgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3hgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3ugradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Ygradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/tuple/control_dependencydgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
Я
Лgradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3-blstm_1/bidirectional_rnn/fw/fw/TensorArray_1Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/fw/TensorArray_1*
source	gradients*
_output_shapes

:: 
▐
Зgradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityVgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3М^gradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/fw/TensorArray_1*
_output_shapes
: 
┴
}gradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3Лgradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV38blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/rangeЗgradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*5
_output_shapes#
!:                  А
█
zgradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOp~^gradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3W^gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
с
Вgradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentity}gradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3{^gradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*У
_classИ
ЕВloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*5
_output_shapes#
!:                  А
Є
Дgradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityVgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3{^gradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 
Щ
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
dtype0*
_output_shapes
: *
valueB
 *    
╙
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterTgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
┴
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeVgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1\gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
я
Ugradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchVgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_6*
T0*
_output_shapes
: : 
┐
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/AddAddWgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Switch:1jgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
т
\gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationRgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Add*
_output_shapes
: *
T0
╓
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitUgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
м
Jgradients/blstm_1/bidirectional_rnn/fw/fw/transpose_grad/InvertPermutationInvertPermutation&blstm_1/bidirectional_rnn/fw/fw/concat*
T0*
_output_shapes
:
э
Bgradients/blstm_1/bidirectional_rnn/fw/fw/transpose_grad/transpose	TransposeВgradients/blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyJgradients/blstm_1/bidirectional_rnn/fw/fw/transpose_grad/InvertPermutation*5
_output_shapes#
!:                  А*
Tperm0*
T0
Я
Лgradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3-blstm_1/bidirectional_rnn/bw/bw/TensorArray_1Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/bw/TensorArray_1*
source	gradients*
_output_shapes

:: 
▐
Зgradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityVgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3М^gradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/bw/TensorArray_1*
_output_shapes
: 
┴
}gradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3Лgradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV38blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/rangeЗgradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*5
_output_shapes#
!:                  А*
element_shape:
█
zgradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOp~^gradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3W^gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
с
Вgradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentity}gradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3{^gradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*5
_output_shapes#
!:                  А*
T0*У
_classИ
ЕВloc:@gradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
Є
Дgradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityVgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3{^gradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 
м
Jgradients/blstm_1/bidirectional_rnn/bw/bw/transpose_grad/InvertPermutationInvertPermutation&blstm_1/bidirectional_rnn/bw/bw/concat*
_output_shapes
:*
T0
э
Bgradients/blstm_1/bidirectional_rnn/bw/bw/transpose_grad/transpose	TransposeВgradients/blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyJgradients/blstm_1/bidirectional_rnn/bw/bw/transpose_grad/InvertPermutation*
T0*5
_output_shapes#
!:                  А*
Tperm0
д
Kgradients/blstm_1/bidirectional_rnn/bw/ReverseSequence_grad/ReverseSequenceReverseSequenceBgradients/blstm_1/bidirectional_rnn/bw/bw/transpose_grad/transposeinputs/Placeholder_2*
seq_dim*

Tlen0*5
_output_shapes#
!:                  А*
	batch_dim *
T0
╔
gradients/AddN_8AddNBgradients/blstm_1/bidirectional_rnn/fw/fw/transpose_grad/transposeKgradients/blstm_1/bidirectional_rnn/bw/ReverseSequence_grad/ReverseSequence*
T0*U
_classK
IGloc:@gradients/blstm_1/bidirectional_rnn/fw/fw/transpose_grad/transpose*
N*5
_output_shapes#
!:                  А
\
gradients/concat_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
o
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
_output_shapes
: *
T0
Ж
gradients/concat_grad/ShapeShape+blstm_0/bidirectional_rnn/fw/fw/transpose_1*
T0*
out_type0*
_output_shapes
:
░
gradients/concat_grad/ShapeNShapeN+blstm_0/bidirectional_rnn/fw/fw/transpose_1blstm_0/ReverseSequence*
T0*
out_type0*
N* 
_output_shapes
::
╢
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1*
N* 
_output_shapes
::
┼
gradients/concat_grad/SliceSlicegradients/AddN_8"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
Index0*
T0*5
_output_shapes#
!:                  А
╦
gradients/concat_grad/Slice_1Slicegradients/AddN_8$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
Index0*
T0*5
_output_shapes#
!:                  А
l
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1
Ё
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/concat_grad/Slice*5
_output_shapes#
!:                  А
Ў
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_1*5
_output_shapes#
!:                  А
░
Lgradients/blstm_0/bidirectional_rnn/fw/fw/transpose_1_grad/InvertPermutationInvertPermutation(blstm_0/bidirectional_rnn/fw/fw/concat_2*
T0*
_output_shapes
:
Ь
Dgradients/blstm_0/bidirectional_rnn/fw/fw/transpose_1_grad/transpose	Transpose.gradients/concat_grad/tuple/control_dependencyLgradients/blstm_0/bidirectional_rnn/fw/fw/transpose_1_grad/InvertPermutation*
T0*5
_output_shapes#
!:                  А*
Tperm0
¤
6gradients/blstm_0/ReverseSequence_grad/ReverseSequenceReverseSequence0gradients/concat_grad/tuple/control_dependency_1inputs/Placeholder_2*
seq_dim*

Tlen0*5
_output_shapes#
!:                  А*
	batch_dim *
T0
┌
ugradients/blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3+blstm_0/bidirectional_rnn/fw/fw/TensorArray,blstm_0/bidirectional_rnn/fw/fw/while/Exit_2*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/fw/TensorArray*
source	gradients*
_output_shapes

:: 
Д
qgradients/blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity,blstm_0/bidirectional_rnn/fw/fw/while/Exit_2v^gradients/blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/fw/TensorArray*
_output_shapes
: 
Ь
{gradients/blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3ugradients/blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV36blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/rangeDgradients/blstm_0/bidirectional_rnn/fw/fw/transpose_1_grad/transposeqgradients/blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
Д
gradients/zeros_like_5	ZerosLike,blstm_0/bidirectional_rnn/fw/fw/while/Exit_3*
T0*(
_output_shapes
:         А
Д
gradients/zeros_like_6	ZerosLike,blstm_0/bidirectional_rnn/fw/fw/while/Exit_4*
T0*(
_output_shapes
:         А
░
Lgradients/blstm_0/bidirectional_rnn/bw/bw/transpose_1_grad/InvertPermutationInvertPermutation(blstm_0/bidirectional_rnn/bw/bw/concat_2*
_output_shapes
:*
T0
д
Dgradients/blstm_0/bidirectional_rnn/bw/bw/transpose_1_grad/transpose	Transpose6gradients/blstm_0/ReverseSequence_grad/ReverseSequenceLgradients/blstm_0/bidirectional_rnn/bw/bw/transpose_1_grad/InvertPermutation*
Tperm0*
T0*5
_output_shapes#
!:                  А
ц
Bgradients/blstm_0/bidirectional_rnn/fw/fw/while/Exit_2_grad/b_exitEnter{gradients/blstm_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
У
Bgradients/blstm_0/bidirectional_rnn/fw/fw/while/Exit_3_grad/b_exitEntergradients/zeros_like_5*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*(
_output_shapes
:         А
У
Bgradients/blstm_0/bidirectional_rnn/fw/fw/while/Exit_4_grad/b_exitEntergradients/zeros_like_6*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*(
_output_shapes
:         А
┌
ugradients/blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3+blstm_0/bidirectional_rnn/bw/bw/TensorArray,blstm_0/bidirectional_rnn/bw/bw/while/Exit_2*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/bw/TensorArray*
source	gradients*
_output_shapes

:: 
Д
qgradients/blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity,blstm_0/bidirectional_rnn/bw/bw/while/Exit_2v^gradients/blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
: 
Ь
{gradients/blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3ugradients/blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV36blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/rangeDgradients/blstm_0/bidirectional_rnn/bw/bw/transpose_1_grad/transposeqgradients/blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
Д
gradients/zeros_like_7	ZerosLike,blstm_0/bidirectional_rnn/bw/bw/while/Exit_3*(
_output_shapes
:         А*
T0
Д
gradients/zeros_like_8	ZerosLike,blstm_0/bidirectional_rnn/bw/bw/while/Exit_4*(
_output_shapes
:         А*
T0
О
Fgradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switchMergeBgradients/blstm_0/bidirectional_rnn/fw/fw/while/Exit_2_grad/b_exitMgradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_2_grad_1/NextIteration*
N*
_output_shapes
: : *
T0
а
Fgradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switchMergeBgradients/blstm_0/bidirectional_rnn/fw/fw/while/Exit_3_grad/b_exitMgradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_3_grad_1/NextIteration*
T0*
N**
_output_shapes
:         А: 
а
Fgradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switchMergeBgradients/blstm_0/bidirectional_rnn/fw/fw/while/Exit_4_grad/b_exitMgradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_4_grad_1/NextIteration*
N**
_output_shapes
:         А: *
T0
ц
Bgradients/blstm_0/bidirectional_rnn/bw/bw/while/Exit_2_grad/b_exitEnter{gradients/blstm_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
У
Bgradients/blstm_0/bidirectional_rnn/bw/bw/while/Exit_3_grad/b_exitEntergradients/zeros_like_7*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*(
_output_shapes
:         А*
T0*
is_constant( 
У
Bgradients/blstm_0/bidirectional_rnn/bw/bw/while/Exit_4_grad/b_exitEntergradients/zeros_like_8*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*(
_output_shapes
:         А
й
Cgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/SwitchSwitchFgradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switchgradients/b_count_10*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: : 
Ы
Mgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_depsNoOpD^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch
Є
Ugradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependencyIdentityCgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/SwitchN^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: 
Ў
Wgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1IdentityEgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch:1N^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: 
═
Cgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3_grad/SwitchSwitchFgradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switchgradients/b_count_10*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*<
_output_shapes*
(:         А:         А
Ы
Mgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_depsNoOpD^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch
Д
Ugradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependencyIdentityCgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3_grad/SwitchN^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*(
_output_shapes
:         А
И
Wgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1IdentityEgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch:1N^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*(
_output_shapes
:         А
═
Cgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4_grad/SwitchSwitchFgradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switchgradients/b_count_10*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch*<
_output_shapes*
(:         А:         А
Ы
Mgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_depsNoOpD^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch
Д
Ugradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependencyIdentityCgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4_grad/SwitchN^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch*(
_output_shapes
:         А
И
Wgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1IdentityEgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch:1N^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch*(
_output_shapes
:         А
О
Fgradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switchMergeBgradients/blstm_0/bidirectional_rnn/bw/bw/while/Exit_2_grad/b_exitMgradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
а
Fgradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switchMergeBgradients/blstm_0/bidirectional_rnn/bw/bw/while/Exit_3_grad/b_exitMgradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_3_grad_1/NextIteration*
T0*
N**
_output_shapes
:         А: 
а
Fgradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switchMergeBgradients/blstm_0/bidirectional_rnn/bw/bw/while/Exit_4_grad/b_exitMgradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_4_grad_1/NextIteration*
T0*
N**
_output_shapes
:         А: 
┴
Agradients/blstm_0/bidirectional_rnn/fw/fw/while/Enter_2_grad/ExitExitUgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
╙
Agradients/blstm_0/bidirectional_rnn/fw/fw/while/Enter_3_grad/ExitExitUgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
╙
Agradients/blstm_0/bidirectional_rnn/fw/fw/while/Enter_4_grad/ExitExitUgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
й
Cgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/SwitchSwitchFgradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switchgradients/b_count_14*
_output_shapes
: : *
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch
Ы
Mgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_depsNoOpD^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch
Є
Ugradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependencyIdentityCgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/SwitchN^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: 
Ў
Wgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1IdentityEgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch:1N^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch
═
Cgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3_grad/SwitchSwitchFgradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switchgradients/b_count_14*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*<
_output_shapes*
(:         А:         А
Ы
Mgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_depsNoOpD^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch
Д
Ugradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependencyIdentityCgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3_grad/SwitchN^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*(
_output_shapes
:         А
И
Wgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1IdentityEgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch:1N^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*(
_output_shapes
:         А
═
Cgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4_grad/SwitchSwitchFgradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switchgradients/b_count_14*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch*<
_output_shapes*
(:         А:         А
Ы
Mgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_depsNoOpD^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch
Д
Ugradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependencyIdentityCgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4_grad/SwitchN^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch*(
_output_shapes
:         А
И
Wgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1IdentityEgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch:1N^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch*(
_output_shapes
:         А
г
Аgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter+blstm_0/bidirectional_rnn/fw/fw/TensorArray*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context
ъ
zgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Аgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterWgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1*H
_class>
<:loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
source	gradients*
_output_shapes

:: 
├
vgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityWgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1{^gradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2
А
pgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*C
_class9
75loc:@blstm_0/bidirectional_rnn/fw/fw/while/Identity_1*
valueB :
         *
dtype0*
_output_shapes
: 
ё
pgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2pgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*C
_class9
75loc:@blstm_0/bidirectional_rnn/fw/fw/while/Identity_1*

stack_name *
_output_shapes
:*
	elem_type0
Г
pgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterpgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
я
vgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2pgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter0blstm_0/bidirectional_rnn/fw/fw/while/Identity_1^gradients/Add_2*
T0*
swap_memory( *
_output_shapes
: 
Ш
{gradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterpgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╗
ugradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2{gradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_2*
_output_shapes
: *
	elem_type0
б
jgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3zgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3ugradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2vgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:         А
╕
igradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpX^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1k^gradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
З
qgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityjgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3j^gradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*}
_classs
qoloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
└
sgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityWgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1j^gradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch
▐
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/ConstConst*C
_class9
75loc:@blstm_0/bidirectional_rnn/fw/fw/while/Identity_3*
valueB :
         *
dtype0*
_output_shapes
: 
н
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_accStackV2Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Const*
	elem_type0*C
_class9
75loc:@blstm_0/bidirectional_rnn/fw/fw/while/Identity_3*

stack_name *
_output_shapes
:
┐
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/EnterEnterNgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
╜
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPushV2StackPushV2Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Enter0blstm_0/bidirectional_rnn/fw/fw/while/Identity_3^gradients/Add_2*
T0*
swap_memory( *(
_output_shapes
:         А
╘
Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2/EnterEnterNgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Й
Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
▌
Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like	ZerosLikeSgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:         А
▄
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/ConstConst*E
_class;
97loc:@blstm_0/bidirectional_rnn/fw/fw/while/GreaterEqual*
valueB :
         *
dtype0*
_output_shapes
: 
з
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_accStackV2Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Const*E
_class;
97loc:@blstm_0/bidirectional_rnn/fw/fw/while/GreaterEqual*

stack_name *
_output_shapes
:*
	elem_type0

╖
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/EnterEnterJgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
з
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2StackPushV2Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter2blstm_0/bidirectional_rnn/fw/fw/while/GreaterEqual^gradients/Add_2*
swap_memory( *
_output_shapes
:*
T0

╠
Ugradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2/EnterEnterJgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
ё
Ogradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2
StackPopV2Ugradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
*
_output_shapes
:
ї
Dgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectSelectOgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Wgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like*
T0*(
_output_shapes
:         А
ў
Fgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1SelectOgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_likeWgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
ц
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_depsNoOpE^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectG^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
Е
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependencyIdentityDgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectO^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*W
_classM
KIloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select
Л
Xgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependency_1IdentityFgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1O^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
▐
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/ConstConst*C
_class9
75loc:@blstm_0/bidirectional_rnn/fw/fw/while/Identity_4*
valueB :
         *
dtype0*
_output_shapes
: 
н
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_accStackV2Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Const*
	elem_type0*C
_class9
75loc:@blstm_0/bidirectional_rnn/fw/fw/while/Identity_4*

stack_name *
_output_shapes
:
┐
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/EnterEnterNgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╜
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPushV2StackPushV2Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Enter0blstm_0/bidirectional_rnn/fw/fw/while/Identity_4^gradients/Add_2*
T0*
swap_memory( *(
_output_shapes
:         А
╘
Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2/EnterEnterNgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Й
Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
▌
Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like	ZerosLikeSgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2*(
_output_shapes
:         А*
T0
ї
Dgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectSelectOgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Wgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like*
T0*(
_output_shapes
:         А
ў
Fgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1SelectOgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_likeWgradients/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
ц
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_depsNoOpE^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectG^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1
Е
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependencyIdentityDgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectO^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/Select*(
_output_shapes
:         А
Л
Xgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency_1IdentityFgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1O^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1*(
_output_shapes
:         А
┴
Agradients/blstm_0/bidirectional_rnn/bw/bw/while/Enter_2_grad/ExitExitUgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency*
_output_shapes
: *
T0
╙
Agradients/blstm_0/bidirectional_rnn/bw/bw/while/Enter_3_grad/ExitExitUgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
╙
Agradients/blstm_0/bidirectional_rnn/bw/bw/while/Enter_4_grad/ExitExitUgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
м
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/EnterEnter%blstm_0/bidirectional_rnn/fw/fw/zeros*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*(
_output_shapes
:         А*
T0*
is_constant(
ц
Fgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like	ZerosLikeLgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/Enter^gradients/Sub_2*
T0*(
_output_shapes
:         А
Л
Bgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/SelectSelectOgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2qgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyFgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like*
T0*(
_output_shapes
:         А
Н
Dgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/Select_1SelectOgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Fgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/zeros_likeqgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
р
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_depsNoOpC^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/SelectE^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/Select_1
¤
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependencyIdentityBgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/SelectM^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/Select*(
_output_shapes
:         А
Г
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency_1IdentityDgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/Select_1M^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/Select_1*(
_output_shapes
:         А
г
Аgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter+blstm_0/bidirectional_rnn/bw/bw/TensorArray*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context
ъ
zgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Аgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterWgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1*H
_class>
<:loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
source	gradients*
_output_shapes

:: 
├
vgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityWgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1{^gradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*H
_class>
<:loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
_output_shapes
: 
А
pgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*C
_class9
75loc:@blstm_0/bidirectional_rnn/bw/bw/while/Identity_1*
valueB :
         *
dtype0*
_output_shapes
: 
ё
pgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2pgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *
_output_shapes
:*
	elem_type0*C
_class9
75loc:@blstm_0/bidirectional_rnn/bw/bw/while/Identity_1
Г
pgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterpgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
я
vgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2pgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter0blstm_0/bidirectional_rnn/bw/bw/while/Identity_1^gradients/Add_3*
T0*
swap_memory( *
_output_shapes
: 
Ш
{gradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterpgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
╗
ugradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2{gradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_3*
_output_shapes
: *
	elem_type0
б
jgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3zgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3ugradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2vgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:         А
╕
igradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpX^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1k^gradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
З
qgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityjgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3j^gradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*}
_classs
qoloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*(
_output_shapes
:         А
└
sgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityWgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1j^gradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: 
▐
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/ConstConst*C
_class9
75loc:@blstm_0/bidirectional_rnn/bw/bw/while/Identity_3*
valueB :
         *
dtype0*
_output_shapes
: 
н
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_accStackV2Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Const*
	elem_type0*C
_class9
75loc:@blstm_0/bidirectional_rnn/bw/bw/while/Identity_3*

stack_name *
_output_shapes
:
┐
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/EnterEnterNgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
╜
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPushV2StackPushV2Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Enter0blstm_0/bidirectional_rnn/bw/bw/while/Identity_3^gradients/Add_3*
T0*
swap_memory( *(
_output_shapes
:         А
╘
Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2/EnterEnterNgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Й
Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2/Enter^gradients/Sub_3*(
_output_shapes
:         А*
	elem_type0
▌
Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like	ZerosLikeSgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:         А
▄
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/ConstConst*
dtype0*
_output_shapes
: *E
_class;
97loc:@blstm_0/bidirectional_rnn/bw/bw/while/GreaterEqual*
valueB :
         
з
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_accStackV2Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Const*
	elem_type0
*E
_class;
97loc:@blstm_0/bidirectional_rnn/bw/bw/while/GreaterEqual*

stack_name *
_output_shapes
:
╖
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/EnterEnterJgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
з
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2StackPushV2Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter2blstm_0/bidirectional_rnn/bw/bw/while/GreaterEqual^gradients/Add_3*
T0
*
swap_memory( *
_output_shapes
:
╠
Ugradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2/EnterEnterJgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
ё
Ogradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2
StackPopV2Ugradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2/Enter^gradients/Sub_3*
_output_shapes
:*
	elem_type0

ї
Dgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectSelectOgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Wgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like*(
_output_shapes
:         А*
T0
ў
Fgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1SelectOgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_likeWgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
ц
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_depsNoOpE^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectG^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1
Е
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependencyIdentityDgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectO^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*W
_classM
KIloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select
Л
Xgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependency_1IdentityFgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1O^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1*(
_output_shapes
:         А
▐
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/ConstConst*C
_class9
75loc:@blstm_0/bidirectional_rnn/bw/bw/while/Identity_4*
valueB :
         *
dtype0*
_output_shapes
: 
н
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_accStackV2Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Const*

stack_name *
_output_shapes
:*
	elem_type0*C
_class9
75loc:@blstm_0/bidirectional_rnn/bw/bw/while/Identity_4
┐
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/EnterEnterNgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
╜
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPushV2StackPushV2Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Enter0blstm_0/bidirectional_rnn/bw/bw/while/Identity_4^gradients/Add_3*
swap_memory( *(
_output_shapes
:         А*
T0
╘
Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2/EnterEnterNgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Й
Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2/Enter^gradients/Sub_3*
	elem_type0*(
_output_shapes
:         А
▌
Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like	ZerosLikeSgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:         А
ї
Dgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectSelectOgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Wgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like*(
_output_shapes
:         А*
T0
ў
Fgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1SelectOgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_likeWgradients/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
ц
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_depsNoOpE^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectG^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1
Е
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependencyIdentityDgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectO^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*W
_classM
KIloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/Select
Л
Xgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency_1IdentityFgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1O^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1*(
_output_shapes
:         А
с
gradients/AddN_9AddNXgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency_1Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency_1*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1*
N*(
_output_shapes
:         А
├
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ShapeShape9blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
┬
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1Shape6blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*
_output_shapes
:*
T0*
out_type0
К
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
ы
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
у
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnter`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
э
fgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterJgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape^gradients/Add_2*
swap_memory( *
_output_shapes
:*
T0
°
kgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnter`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Я
egradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2kgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
О
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
ё
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
ч
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enterbgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
є
hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1^gradients/Add_2*
T0*
swap_memory( *
_output_shapes
:
№
mgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterbgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
г
ggradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2mgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_2*
	elem_type0*
_output_shapes
:
А
Zgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsegradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2ggradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
ф
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*
valueB :
         
│
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_accStackV2Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Const*I
_class?
=;loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
┐
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/EnterEnterNgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
├
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Enter6blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1^gradients/Add_2*
T0*
swap_memory( *(
_output_shapes
:         А
╘
Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterNgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Й
Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
щ
Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/MulMulgradients/AddN_9Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*(
_output_shapes
:         А
╡
Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/SumSumHgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/MulZgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╔
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ReshapeReshapeHgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Sumegradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:         А*
T0*
Tshape0
щ
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*L
_classB
@>loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2*
valueB :
         *
dtype0*
_output_shapes
: 
║
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Const*L
_classB
@>loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0
├
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterPgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╩
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Enter9blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2^gradients/Add_2*
swap_memory( *(
_output_shapes
:         А*
T0
╪
[gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterPgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Н
Ugradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2[gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
э
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1MulUgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2gradients/AddN_9*
T0*(
_output_shapes
:         А
╗
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Sum_1SumJgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1\gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╧
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1ReshapeJgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Sum_1ggradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*(
_output_shapes
:         А*
T0*
Tshape0
¤
Ugradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/group_depsNoOpM^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ReshapeO^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1
г
]gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityLgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ReshapeV^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape
й
_gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityNgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1V^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1*(
_output_shapes
:         А
Ї
Mgradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_2_grad_1/NextIterationNextIterationsgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
м
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/EnterEnter%blstm_0/bidirectional_rnn/bw/bw/zeros*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*(
_output_shapes
:         А
ц
Fgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like	ZerosLikeLgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/Enter^gradients/Sub_3*
T0*(
_output_shapes
:         А
Л
Bgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/SelectSelectOgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2qgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyFgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like*
T0*(
_output_shapes
:         А
Н
Dgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/Select_1SelectOgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Fgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/zeros_likeqgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
р
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_depsNoOpC^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/SelectE^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/Select_1
¤
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependencyIdentityBgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/SelectM^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/Select*(
_output_shapes
:         А
Г
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency_1IdentityDgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/Select_1M^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/Select_1*(
_output_shapes
:         А
╠
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradUgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2]gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
├
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradSgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2_gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
т
gradients/AddN_10AddNXgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency_1Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency_1*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1*
N*(
_output_shapes
:         А
├
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ShapeShape9blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
┬
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1Shape6blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*
T0*
out_type0*
_output_shapes
:
К
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
ы
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*
	elem_type0*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape*

stack_name *
_output_shapes
:
у
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnter`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
э
fgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterJgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape^gradients/Add_3*
T0*
swap_memory( *
_output_shapes
:
°
kgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnter`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Я
egradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2kgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_3*
	elem_type0*
_output_shapes
:
О
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
ё
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
ч
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enterbgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
є
hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1^gradients/Add_3*
T0*
swap_memory( *
_output_shapes
:
№
mgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterbgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
г
ggradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2mgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_3*
_output_shapes
:*
	elem_type0
А
Zgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsegradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2ggradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
ф
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/ConstConst*I
_class?
=;loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*
valueB :
         *
dtype0*
_output_shapes
: 
│
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_accStackV2Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Const*I
_class?
=;loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
┐
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/EnterEnterNgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
├
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Enter6blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1^gradients/Add_3*
T0*
swap_memory( *(
_output_shapes
:         А
╘
Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterNgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Й
Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_3*(
_output_shapes
:         А*
	elem_type0
ъ
Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/MulMulgradients/AddN_10Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*(
_output_shapes
:         А
╡
Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/SumSumHgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/MulZgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╔
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ReshapeReshapeHgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Sumegradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
щ
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *L
_classB
@>loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2*
valueB :
         
║
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*L
_classB
@>loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2
├
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterPgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
╩
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Enter9blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2^gradients/Add_3*
swap_memory( *(
_output_shapes
:         А*
T0
╪
[gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterPgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Н
Ugradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2[gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_3*(
_output_shapes
:         А*
	elem_type0
ю
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1MulUgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2gradients/AddN_10*
T0*(
_output_shapes
:         А
╗
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Sum_1SumJgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1\gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╧
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1ReshapeJgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Sum_1ggradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
¤
Ugradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/group_depsNoOpM^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ReshapeO^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1
г
]gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityLgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ReshapeV^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape*(
_output_shapes
:         А
й
_gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityNgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1V^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1*(
_output_shapes
:         А
Ї
Mgradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_2_grad_1/NextIterationNextIterationsgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
┌
gradients/AddN_11AddNXgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependency_1Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1_grad/TanhGrad*(
_output_shapes
:         А*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1*
N
╜
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ShapeShape3blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:
┴
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1Shape5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
К
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape*
valueB :
         
ы
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*
_output_shapes
:*
	elem_type0*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape*

stack_name 
у
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnter`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
э
fgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterJgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape^gradients/Add_2*
T0*
swap_memory( *
_output_shapes
:
°
kgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Я
egradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2kgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
О
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1*
valueB :
         
ё
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1
ч
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enterbgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
є
hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1^gradients/Add_2*
swap_memory( *
_output_shapes
:*
T0
№
mgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterbgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
г
ggradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2mgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
А
Zgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsegradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2ggradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
■
Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/SumSumgradients/AddN_11Zgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╔
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ReshapeReshapeHgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Sumegradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
В
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_11\gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╧
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1ReshapeJgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Sum_1ggradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
¤
Ugradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/group_depsNoOpM^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ReshapeO^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1
г
]gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependencyIdentityLgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ReshapeV^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape*(
_output_shapes
:         А
й
_gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependency_1IdentityNgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1V^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1*(
_output_shapes
:         А
╠
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradUgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2]gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
├
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradSgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2_gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
┐
Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ShapeShape7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid*
_output_shapes
:*
T0*
out_type0
║
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1Shape0blstm_0/bidirectional_rnn/fw/fw/while/Identity_3*
T0*
out_type0*
_output_shapes
:
Ж
^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*[
_classQ
OMloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
х
^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*[
_classQ
OMloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
▀
^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnter^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
ч
dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterHgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape^gradients/Add_2*
T0*
swap_memory( *
_output_shapes
:
Ї
igradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Ы
cgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
К
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
ы
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
у
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enter`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
э
fgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1^gradients/Add_2*
T0*
swap_memory( *
_output_shapes
:
°
kgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Я
egradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2kgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
·
Xgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2egradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
┤
Fgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/MulMul]gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependencySgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:         А
п
Fgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/SumSumFgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/MulXgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
├
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ReshapeReshapeFgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Sumcgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
х
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid*
valueB :
         
┤
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_accStackV2Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Const*
	elem_type0*J
_class@
><loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:
┐
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/EnterEnterNgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
─
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Enter7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid^gradients/Add_2*
swap_memory( *(
_output_shapes
:         А*
T0
╘
Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterNgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Й
Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
╢
Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1MulSgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2]gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
╡
Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Sum_1SumHgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1Zgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╔
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1ReshapeHgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Sum_1egradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
ў
Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/group_depsNoOpK^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ReshapeM^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1
Ы
[gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/control_dependencyIdentityJgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ReshapeT^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape
б
]gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/control_dependency_1IdentityLgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1T^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1*(
_output_shapes
:         А
├
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ShapeShape9blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
└
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1Shape4blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
К
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
ы
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape
у
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnter`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
э
fgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterJgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape^gradients/Add_2*
T0*
swap_memory( *
_output_shapes
:
°
kgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Я
egradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2kgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
О
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
ё
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1*

stack_name *
_output_shapes
:
ч
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enterbgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
є
hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1^gradients/Add_2*
swap_memory( *
_output_shapes
:*
T0
№
mgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterbgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
г
ggradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2mgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
А
Zgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsegradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2ggradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
т
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *G
_class=
;9loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh*
valueB :
         
▒
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_accStackV2Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*G
_class=
;9loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh
┐
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/EnterEnterNgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
┴
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Enter4blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh^gradients/Add_2*
T0*
swap_memory( *(
_output_shapes
:         А
╘
Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterNgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Й
Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
╕
Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/MulMul_gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependency_1Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*(
_output_shapes
:         А
╡
Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/SumSumHgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/MulZgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╔
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ReshapeReshapeHgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Sumegradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
щ
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*L
_classB
@>loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1*
valueB :
         *
dtype0*
_output_shapes
: 
║
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Const*
	elem_type0*L
_classB
@>loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:
├
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterPgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╩
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Enter9blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1^gradients/Add_2*
T0*
swap_memory( *(
_output_shapes
:         А
╪
[gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterPgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Н
Ugradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2[gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
╝
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1MulUgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2_gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
╗
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Sum_1SumJgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1\gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╧
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1ReshapeJgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Sum_1ggradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
¤
Ugradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/group_depsNoOpM^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ReshapeO^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1
г
]gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityLgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ReshapeV^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape*(
_output_shapes
:         А
й
_gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityNgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1V^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*a
_classW
USloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1
┌
gradients/AddN_12AddNXgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependency_1Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1_grad/TanhGrad*
T0*Y
_classO
MKloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1*
N*(
_output_shapes
:         А
╜
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ShapeShape3blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:
┴
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1Shape5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
К
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape*
valueB :
         
ы
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
у
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnter`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
э
fgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterJgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape^gradients/Add_3*
swap_memory( *
_output_shapes
:*
T0
°
kgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Я
egradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2kgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_3*
_output_shapes
:*
	elem_type0
О
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1*
valueB :
         
ё
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1
ч
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enterbgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
є
hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1^gradients/Add_3*
swap_memory( *
_output_shapes
:*
T0
№
mgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterbgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
г
ggradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2mgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_3*
_output_shapes
:*
	elem_type0
А
Zgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsegradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2ggradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
■
Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/SumSumgradients/AddN_12Zgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╔
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ReshapeReshapeHgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Sumegradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
В
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_12\gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╧
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1ReshapeJgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Sum_1ggradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
¤
Ugradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/group_depsNoOpM^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ReshapeO^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1
г
]gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependencyIdentityLgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ReshapeV^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape*(
_output_shapes
:         А
й
_gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependency_1IdentityNgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1V^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1*(
_output_shapes
:         А
╞
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradSgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2[gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
х
gradients/AddN_13AddNVgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependency]gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select*
N*(
_output_shapes
:         А
╠
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradUgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2]gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
┴
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_grad/TanhGradTanhGradSgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2_gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
┐
Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ShapeShape7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
║
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1Shape0blstm_0/bidirectional_rnn/bw/bw/while/Identity_3*
T0*
out_type0*
_output_shapes
:
Ж
^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*[
_classQ
OMloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
х
^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*[
_classQ
OMloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
▀
^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnter^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
ч
dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterHgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape^gradients/Add_3*
T0*
swap_memory( *
_output_shapes
:
Ї
igradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Ы
cgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_3*
_output_shapes
:*
	elem_type0
К
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1*
valueB :
         *
dtype0*
_output_shapes
: 
ы
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
у
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enter`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
э
fgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1^gradients/Add_3*
swap_memory( *
_output_shapes
:*
T0
°
kgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Я
egradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2kgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_3*
	elem_type0*
_output_shapes
:
·
Xgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2egradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
┤
Fgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/MulMul]gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependencySgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2*(
_output_shapes
:         А*
T0
п
Fgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/SumSumFgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/MulXgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
├
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ReshapeReshapeFgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Sumcgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
х
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/ConstConst*J
_class@
><loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid*
valueB :
         *
dtype0*
_output_shapes
: 
┤
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_accStackV2Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Const*J
_class@
><loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
┐
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/EnterEnterNgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
─
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Enter7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid^gradients/Add_3*
T0*
swap_memory( *(
_output_shapes
:         А
╘
Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterNgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Й
Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_3*(
_output_shapes
:         А*
	elem_type0
╢
Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1MulSgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2]gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
╡
Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Sum_1SumHgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1Zgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╔
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1ReshapeHgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Sum_1egradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
ў
Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/group_depsNoOpK^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ReshapeM^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1
Ы
[gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/control_dependencyIdentityJgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ReshapeT^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape*(
_output_shapes
:         А
б
]gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/control_dependency_1IdentityLgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1T^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1*(
_output_shapes
:         А
├
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ShapeShape9blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
└
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1Shape4blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*
_output_shapes
:*
T0*
out_type0
К
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
ы
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape
у
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnter`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
э
fgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterJgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape^gradients/Add_3*
swap_memory( *
_output_shapes
:*
T0
°
kgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Я
egradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2kgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_3*
	elem_type0*
_output_shapes
:
О
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1*
valueB :
         
ё
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
ч
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enterbgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
є
hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1^gradients/Add_3*
T0*
swap_memory( *
_output_shapes
:
№
mgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterbgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
г
ggradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2mgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_3*
_output_shapes
:*
	elem_type0
А
Zgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsegradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2ggradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
т
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/ConstConst*G
_class=
;9loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*
valueB :
         *
dtype0*
_output_shapes
: 
▒
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_accStackV2Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Const*
	elem_type0*G
_class=
;9loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*

stack_name *
_output_shapes
:
┐
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/EnterEnterNgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
┴
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Enter4blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh^gradients/Add_3*
swap_memory( *(
_output_shapes
:         А*
T0
╘
Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterNgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Й
Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_3*(
_output_shapes
:         А*
	elem_type0
╕
Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/MulMul_gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependency_1Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*(
_output_shapes
:         А
╡
Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/SumSumHgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/MulZgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╔
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ReshapeReshapeHgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Sumegradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:         А*
T0*
Tshape0
щ
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *L
_classB
@>loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1*
valueB :
         
║
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*L
_classB
@>loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1
├
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterPgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
╩
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Enter9blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1^gradients/Add_3*
swap_memory( *(
_output_shapes
:         А*
T0
╪
[gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterPgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Н
Ugradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2[gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_3*
	elem_type0*(
_output_shapes
:         А
╝
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1MulUgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2_gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
╗
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Sum_1SumJgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1\gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╧
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1ReshapeJgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Sum_1ggradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:         А
¤
Ugradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/group_depsNoOpM^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ReshapeO^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1
г
]gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityLgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ReshapeV^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape*(
_output_shapes
:         А
й
_gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityNgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1V^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*a
_classW
USloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1
┐
Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ShapeShape7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:
Я
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape_1Const^gradients/Sub_2*
valueB *
dtype0*
_output_shapes
: 
Ж
^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape*
valueB :
         
х
^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*[
_classQ
OMloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
▀
^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnter^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
ч
dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterHgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape^gradients/Add_2*
T0*
swap_memory( *
_output_shapes
:
Ї
igradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
Ы
cgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
	elem_type0*
_output_shapes
:
▀
Xgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╗
Fgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/SumSumRgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_grad/SigmoidGradXgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
├
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ReshapeReshapeFgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Sumcgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
┐
Hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Sum_1SumRgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_grad/SigmoidGradZgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ь
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1ReshapeHgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Sum_1Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
ў
Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/group_depsNoOpK^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ReshapeM^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1
Ы
[gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/control_dependencyIdentityJgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ReshapeT^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/group_deps*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape*(
_output_shapes
:         А*
T0
П
]gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/control_dependency_1IdentityLgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1T^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1*
_output_shapes
: 
д
Mgradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_13*
T0*(
_output_shapes
:         А
╞
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradSgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2[gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
х
gradients/AddN_14AddNVgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependency]gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select*
N*(
_output_shapes
:         А
╠
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradUgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2]gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
┴
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_grad/TanhGradTanhGradSgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2_gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
е
Qgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat/ConstConst^gradients/Sub_2*
value	B :*
dtype0*
_output_shapes
: 
╔
Kgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concatConcatV2Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradLgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_grad/TanhGrad[gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/control_dependencyTgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradQgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat/Const*
T0*
N*(
_output_shapes
:         А*

Tidx0
┐
Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ShapeShape7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:
Я
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape_1Const^gradients/Sub_3*
valueB *
dtype0*
_output_shapes
: 
Ж
^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*[
_classQ
OMloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape*
valueB :
         *
dtype0*
_output_shapes
: 
х
^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*
_output_shapes
:*
	elem_type0*[
_classQ
OMloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape*

stack_name 
▀
^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnter^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
ч
dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterHgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape^gradients/Add_3*
swap_memory( *
_output_shapes
:*
T0
Ї
igradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Ы
cgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_3*
	elem_type0*
_output_shapes
:
▀
Xgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╗
Fgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/SumSumRgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_grad/SigmoidGradXgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
├
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ReshapeReshapeFgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Sumcgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:         А
┐
Hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Sum_1SumRgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_grad/SigmoidGradZgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ь
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1ReshapeHgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Sum_1Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ў
Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/group_depsNoOpK^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ReshapeM^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1
Ы
[gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/control_dependencyIdentityJgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ReshapeT^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape*(
_output_shapes
:         А
П
]gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/control_dependency_1IdentityLgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1T^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1*
_output_shapes
: 
д
Mgradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_14*
T0*(
_output_shapes
:         А
ы
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradKgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:А
В
Wgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpS^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGradL^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat
е
_gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityKgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concatX^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat*(
_output_shapes
:         А
и
agradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityRgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGradX^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
е
Qgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat/ConstConst^gradients/Sub_3*
value	B :*
dtype0*
_output_shapes
: 
╔
Kgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concatConcatV2Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradLgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_grad/TanhGrad[gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/control_dependencyTgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradQgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat/Const*
T0*
N*(
_output_shapes
:         А*

Tidx0
╖
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul/EnterEnter2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context* 
_output_shapes
:
зА
ф
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMulMatMul_gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyRgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul/Enter*
T0*
transpose_a( *(
_output_shapes
:         з*
transpose_b(
ъ
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*I
_class?
=;loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat*
valueB :
         *
dtype0*
_output_shapes
: 
┐
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*I
_class?
=;loc:@blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat*

stack_name *
_output_shapes
:
╦
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterTgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
╧
Zgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Enter6blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat^gradients/Add_2*
T0*
swap_memory( *(
_output_shapes
:         з
р
_gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterTgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
Х
Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2_gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_2*
	elem_type0*(
_output_shapes
:         з
х
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1MatMulYgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2_gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:
зА
■
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/group_depsNoOpM^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMulO^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1
е
^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityLgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMulW^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul*(
_output_shapes
:         з
г
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityNgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1W^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
зА
б
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueBА*    *
dtype0*
_output_shapes	
:А
╘
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterRgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes	
:А
└
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeTgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1Zgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:А: 
Ў
Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchTgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_10*
T0*"
_output_shapes
:А:А
╖
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/AddAddUgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/Switch:1agradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:А
у
Zgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationPgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:А
╫
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitSgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:А*
T0
ы
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradKgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:А*
T0
В
Wgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpS^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGradL^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat
е
_gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityKgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concatX^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*^
_classT
RPloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat
и
agradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityRgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGradX^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*e
_class[
YWloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGrad
Я
Kgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConstConst^gradients/Sub_2*
value	B :*
dtype0*
_output_shapes
: 
Ю
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/RankConst^gradients/Sub_2*
dtype0*
_output_shapes
: *
value	B :
П
Igradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/modFloorModKgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConstJgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
╝
Kgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeShape1blstm_0/bidirectional_rnn/fw/fw/while/dropout/mul*
_output_shapes
:*
T0*
out_type0
у
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/ConstConst*D
_class:
86loc:@blstm_0/bidirectional_rnn/fw/fw/while/dropout/mul*
valueB :
         *
dtype0*
_output_shapes
: 
╢
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_accStackV2Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Const*D
_class:
86loc:@blstm_0/bidirectional_rnn/fw/fw/while/dropout/mul*

stack_name *
_output_shapes
:*
	elem_type0
╟
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/EnterEnterRgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc*C

frame_name53blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(*
parallel_iterations 
┼
Xgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Enter1blstm_0/bidirectional_rnn/fw/fw/while/dropout/mul^gradients/Add_2*
swap_memory( *'
_output_shapes
:         '*
T0
°
gradients/NextIteration_4NextIterationgradients/Add_2Q^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2U^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPushV2U^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPushV2w^gradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2[^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2g^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2i^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1e^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2Y^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPushV2g^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2i^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1U^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPushV2W^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2g^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2i^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1U^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPushV2W^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2e^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2g^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1U^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPushV2*
T0*
_output_shapes
: 
▄
]gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterRgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0
Р
Wgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2]gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_2*'
_output_shapes
:         '*
	elem_type0
Л
qgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerP^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2T^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2T^gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2v^gradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Z^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2f^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2h^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1d^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2X^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2f^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2h^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1T^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2V^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2f^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2h^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1T^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2V^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2d^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2f^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1T^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
╨
gradients/NextIteration_5NextIterationgradients/Sub_2r^gradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
╚
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeNShapeNWgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2*
out_type0*
N* 
_output_shapes
::*
T0
Ў
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConcatOffsetConcatOffsetIgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/modLgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeNNgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
Х
Kgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/SliceSlice^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependencyRgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConcatOffsetLgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN*'
_output_shapes
:         '*
Index0*
T0
Ь
Mgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1Slice^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependencyTgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConcatOffset:1Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:         А
№
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/group_depsNoOpL^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/SliceN^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1
в
^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/control_dependencyIdentityKgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/SliceW^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice*'
_output_shapes
:         '
й
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/control_dependency_1IdentityMgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1W^gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/group_deps*`
_classV
TRloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1*(
_output_shapes
:         А*
T0
к
Qgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
зА*    *
dtype0* 
_output_shapes
:
зА
╫
Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterQgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/fw/fw/while/while_context* 
_output_shapes
:
зА
┬
Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeSgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_1Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
зА: 
■
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchSgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_10*
T0*,
_output_shapes
:
зА:
зА
╣
Ogradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/AddAddTgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/Switch:1`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
зА*
T0
ц
Ygradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationOgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
зА
┌
Sgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitRgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
зА*
T0
╖
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul/EnterEnter2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context* 
_output_shapes
:
зА
ф
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMulMatMul_gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyRgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:         з
ъ
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*I
_class?
=;loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat*
valueB :
         *
dtype0*
_output_shapes
: 
┐
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*I
_class?
=;loc:@blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat*

stack_name *
_output_shapes
:
╦
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterTgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(*
parallel_iterations 
╧
Zgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Enter6blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat^gradients/Add_3*
swap_memory( *(
_output_shapes
:         з*
T0
р
_gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterTgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Х
Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2_gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_3*(
_output_shapes
:         з*
	elem_type0
х
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1MatMulYgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2_gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_a(* 
_output_shapes
:
зА*
transpose_b( *
T0
■
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/group_depsNoOpM^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMulO^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1
е
^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityLgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMulW^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul*(
_output_shapes
:         з
г
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityNgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1W^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
зА
б
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueBА*    *
dtype0*
_output_shapes	
:А
╘
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterRgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes	
:А
└
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeTgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1Zgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:А: 
Ў
Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchTgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_14*
T0*"
_output_shapes
:А:А
╖
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/AddAddUgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/Switch:1agradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:А*
T0
у
Zgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationPgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:А*
T0
╫
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitSgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:А*
T0
ш
gradients/AddN_15AddNVgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/Select*
N*(
_output_shapes
:         А
Я
Kgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConstConst^gradients/Sub_3*
_output_shapes
: *
value	B :*
dtype0
Ю
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/RankConst^gradients/Sub_3*
dtype0*
_output_shapes
: *
value	B :
П
Igradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/modFloorModKgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConstJgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
╝
Kgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeShape1blstm_0/bidirectional_rnn/bw/bw/while/dropout/mul*
T0*
out_type0*
_output_shapes
:
у
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/ConstConst*D
_class:
86loc:@blstm_0/bidirectional_rnn/bw/bw/while/dropout/mul*
valueB :
         *
dtype0*
_output_shapes
: 
╢
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_accStackV2Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Const*D
_class:
86loc:@blstm_0/bidirectional_rnn/bw/bw/while/dropout/mul*

stack_name *
_output_shapes
:*
	elem_type0
╟
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/EnterEnterRgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *C

frame_name53blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
┼
Xgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Enter1blstm_0/bidirectional_rnn/bw/bw/while/dropout/mul^gradients/Add_3*
T0*
swap_memory( *'
_output_shapes
:         '
°
gradients/NextIteration_6NextIterationgradients/Add_3Q^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2U^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPushV2U^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPushV2w^gradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2[^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2g^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2i^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1e^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2Y^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPushV2g^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2i^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1U^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPushV2W^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2g^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2i^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1U^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPushV2W^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2e^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2g^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1U^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPushV2*
_output_shapes
: *
T0
▄
]gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterRgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
Р
Wgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2]gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_3*'
_output_shapes
:         '*
	elem_type0
Л
qgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerP^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2T^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2T^gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2v^gradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Z^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2f^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2h^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1d^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2X^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2f^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2h^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1T^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2V^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2f^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2h^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1T^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2V^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2d^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2f^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1T^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
╨
gradients/NextIteration_7NextIterationgradients/Sub_3r^gradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
╚
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeNShapeNWgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2*
N* 
_output_shapes
::*
T0*
out_type0
Ў
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConcatOffsetConcatOffsetIgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/modLgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeNNgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
Х
Kgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/SliceSlice^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependencyRgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConcatOffsetLgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN*
Index0*
T0*'
_output_shapes
:         '
Ь
Mgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1Slice^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependencyTgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConcatOffset:1Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:         А
№
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/group_depsNoOpL^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/SliceN^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1
в
^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/control_dependencyIdentityKgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/SliceW^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice*'
_output_shapes
:         '
й
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/control_dependency_1IdentityMgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1W^gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1*(
_output_shapes
:         А
к
Qgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
зА*    *
dtype0* 
_output_shapes
:
зА
╫
Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterQgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *M

frame_name?=gradients/blstm_0/bidirectional_rnn/bw/bw/while/while_context* 
_output_shapes
:
зА
┬
Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeSgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_1Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/NextIteration*
N*"
_output_shapes
:
зА: *
T0
■
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchSgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_14*,
_output_shapes
:
зА:
зА*
T0
╣
Ogradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/AddAddTgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/Switch:1`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
зА*
T0
ц
Ygradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationOgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
зА*
T0
┌
Sgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitRgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
зА*
T0
д
Mgradients/blstm_0/bidirectional_rnn/fw/fw/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_15*(
_output_shapes
:         А*
T0
ш
gradients/AddN_16AddNVgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/control_dependency_1*
N*(
_output_shapes
:         А*
T0*W
_classM
KIloc:@gradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/Select
д
Mgradients/blstm_0/bidirectional_rnn/bw/bw/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_16*
T0*(
_output_shapes
:         А
Ю
beta1_power/initial_valueConst*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
п
beta1_power
VariableV2*
shared_name *>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
╬
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes
: *
use_locking(
К
beta1_power/readIdentitybeta1_power*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes
: 
Ю
beta2_power/initial_valueConst*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
п
beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
	container *
shape: 
╬
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes
: *
use_locking(
К
beta2_power/readIdentitybeta2_power*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes
: 
ч
Tblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
valueB"з      *
dtype0*
_output_shapes
:
╤
Jblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
э
Dblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zerosFillTblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorJblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
зА*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*

index_type0
ь
2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
зА*
shared_name *@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
	container *
shape:
зА
╙
9blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam/AssignAssign2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/AdamDblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
зА*
use_locking(*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel
ф
7blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam/readIdentity2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam* 
_output_shapes
:
зА*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel
щ
Vblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
valueB"з      *
dtype0*
_output_shapes
:
╙
Lblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
є
Fblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zerosFillVblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorLblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*

index_type0* 
_output_shapes
:
зА
ю
4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1
VariableV2*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
	container *
shape:
зА*
dtype0* 
_output_shapes
:
зА*
shared_name 
┘
;blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/AssignAssign4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1Fblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА
ш
9blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/readIdentity4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel* 
_output_shapes
:
зА
╤
Bblstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
valueBА*    
▐
0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
	container 
╞
7blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam/AssignAssign0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/AdamBblstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zeros*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
┘
5blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam/readIdentity0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
_output_shapes	
:А
╙
Dblstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
р
2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1
VariableV2*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
╠
9blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/AssignAssign2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1Dblstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zeros*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
▌
7blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/readIdentity2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
_output_shapes	
:А
ч
Tblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
valueB"з      *
dtype0*
_output_shapes
:
╤
Jblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
э
Dblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zerosFillTblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorJblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
зА*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*

index_type0
ь
2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam
VariableV2*
	container *
shape:
зА*
dtype0* 
_output_shapes
:
зА*
shared_name *@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel
╙
9blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam/AssignAssign2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/AdamDblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА
ф
7blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam/readIdentity2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
зА
щ
Vblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
valueB"з      
╙
Lblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
є
Fblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zerosFillVblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorLblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*

index_type0* 
_output_shapes
:
зА
ю
4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1
VariableV2*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
	container *
shape:
зА*
dtype0* 
_output_shapes
:
зА*
shared_name 
┘
;blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/AssignAssign4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1Fblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА
ш
9blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/readIdentity4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
зА
╤
Bblstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
▐
0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
	container *
shape:А
╞
7blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam/AssignAssign0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/AdamBblstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
┘
5blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam/readIdentity0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes	
:А
╙
Dblstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
р
2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
	container *
shape:А
╠
9blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/AssignAssign2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1Dblstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias
▌
7blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/readIdentity2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes	
:А
ч
Tblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
valueB"А     *
dtype0*
_output_shapes
:
╤
Jblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
э
Dblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zerosFillTblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorJblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*

index_type0* 
_output_shapes
:
АА
ь
2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam
VariableV2*
shared_name *@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
╙
9blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/AssignAssign2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/AdamDblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel
ф
7blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/readIdentity2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel* 
_output_shapes
:
АА
щ
Vblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
valueB"А     *
dtype0*
_output_shapes
:
╙
Lblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
valueB
 *    
є
Fblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zerosFillVblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorLblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*

index_type0* 
_output_shapes
:
АА
ю
4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
	container *
shape:
АА
┘
;blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/AssignAssign4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1Fblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
ш
9blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/readIdentity4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel* 
_output_shapes
:
АА
╤
Bblstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
valueBА*    
▐
0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
	container 
╞
7blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam/AssignAssign0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/AdamBblstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
┘
5blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam/readIdentity0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam*
_output_shapes	
:А*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias
╙
Dblstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
р
2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1
VariableV2*
shared_name *>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
╠
9blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/AssignAssign2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1Dblstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
▌
7blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/readIdentity2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1*
_output_shapes	
:А*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias
ч
Tblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
valueB"А     *
dtype0*
_output_shapes
:
╤
Jblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
э
Dblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zerosFillTblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorJblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*

index_type0* 
_output_shapes
:
АА
ь
2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
	container *
shape:
АА
╙
9blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/AssignAssign2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/AdamDblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
ф
7blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/readIdentity2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
АА
щ
Vblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
valueB"А     *
dtype0*
_output_shapes
:
╙
Lblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
є
Fblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zerosFillVblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorLblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*

index_type0* 
_output_shapes
:
АА
ю
4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
	container *
shape:
АА
┘
;blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/AssignAssign4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1Fblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
ш
9blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/readIdentity4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
АА
╤
Bblstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
valueBА*    
▐
0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
	container *
shape:А
╞
7blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam/AssignAssign0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/AdamBblstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
┘
5blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam/readIdentity0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam*
_output_shapes	
:А*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias
╙
Dblstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
р
2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
	container 
╠
9blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/AssignAssign2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1Dblstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
▌
7blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/readIdentity2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes	
:А
е
3dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@dense/kernel*
valueB"   @   *
dtype0*
_output_shapes
:
П
)dense/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ш
#dense/kernel/Adam/Initializer/zerosFill3dense/kernel/Adam/Initializer/zeros/shape_as_tensor)dense/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@dense/kernel*

index_type0*
_output_shapes
:	А@
и
dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	А@*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	А@
╬
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	А@*
use_locking(*
T0*
_class
loc:@dense/kernel
А
dense/kernel/Adam/readIdentitydense/kernel/Adam*
_output_shapes
:	А@*
T0*
_class
loc:@dense/kernel
з
5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"   @   
С
+dense/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *    
ю
%dense/kernel/Adam_1/Initializer/zerosFill5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor+dense/kernel/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@dense/kernel*

index_type0*
_output_shapes
:	А@
к
dense/kernel/Adam_1
VariableV2*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	А@*
dtype0*
_output_shapes
:	А@
╘
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	А@
Д
dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	А@
Н
!dense/bias/Adam/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Ъ
dense/bias/Adam
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@dense/bias
┴
dense/bias/Adam/AssignAssigndense/bias/Adam!dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:@
u
dense/bias/Adam/readIdentitydense/bias/Adam*
_output_shapes
:@*
T0*
_class
loc:@dense/bias
П
#dense/bias/Adam_1/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Ь
dense/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
╟
dense/bias/Adam_1/AssignAssigndense/bias/Adam_1#dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@dense/bias
y
dense/bias/Adam_1/readIdentitydense/bias/Adam_1*
T0*
_class
loc:@dense/bias*
_output_shapes
:@
Э
%dense_1/kernel/Adam/Initializer/zerosConst*!
_class
loc:@dense_1/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
к
dense_1/kernel/Adam
VariableV2*!
_class
loc:@dense_1/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
╒
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adam%dense_1/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*!
_class
loc:@dense_1/kernel
Е
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*
_output_shapes

:@*
T0*!
_class
loc:@dense_1/kernel
Я
'dense_1/kernel/Adam_1/Initializer/zerosConst*!
_class
loc:@dense_1/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
м
dense_1/kernel/Adam_1
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *!
_class
loc:@dense_1/kernel*
	container 
█
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1'dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:@
Й
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*
_output_shapes

:@*
T0*!
_class
loc:@dense_1/kernel
С
#dense_1/bias/Adam/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB*    *
dtype0*
_output_shapes
:
Ю
dense_1/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@dense_1/bias*
	container *
shape:
╔
dense_1/bias/Adam/AssignAssigndense_1/bias/Adam#dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
{
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
_output_shapes
:*
T0*
_class
loc:@dense_1/bias
У
%dense_1/bias/Adam_1/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB*    *
dtype0*
_output_shapes
:
а
dense_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@dense_1/bias*
	container *
shape:
╧
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1%dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_1/bias

dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *╖Q8
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
░
CAdam/update_blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdam	ApplyAdam-blstm_0/bidirectional_rnn/fw/lstm_cell/kernel2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
use_nesterov( * 
_output_shapes
:
зА*
use_locking( *
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel
в
AAdam/update_blstm_0/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdam	ApplyAdam+blstm_0/bidirectional_rnn/fw/lstm_cell/bias0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonTgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_nesterov( *
_output_shapes	
:А*
use_locking( *
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias
░
CAdam/update_blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdam	ApplyAdam-blstm_0/bidirectional_rnn/bw/lstm_cell/kernel2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
use_nesterov( * 
_output_shapes
:
зА*
use_locking( *
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel
в
AAdam/update_blstm_0/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdam	ApplyAdam+blstm_0/bidirectional_rnn/bw/lstm_cell/bias0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonTgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_nesterov( *
_output_shapes	
:А*
use_locking( *
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias
░
CAdam/update_blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdam	ApplyAdam-blstm_1/bidirectional_rnn/fw/lstm_cell/kernel2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
use_nesterov( * 
_output_shapes
:
АА
в
AAdam/update_blstm_1/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdam	ApplyAdam+blstm_1/bidirectional_rnn/fw/lstm_cell/bias0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonTgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
use_nesterov( *
_output_shapes	
:А
░
CAdam/update_blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdam	ApplyAdam-blstm_1/bidirectional_rnn/bw/lstm_cell/kernel2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
use_nesterov( * 
_output_shapes
:
АА*
use_locking( 
в
AAdam/update_blstm_1/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdam	ApplyAdam+blstm_1/bidirectional_rnn/bw/lstm_cell/bias0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonTgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
use_nesterov( *
_output_shapes	
:А
є
"Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/dense/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
use_nesterov( *
_output_shapes
:	А@
х
 Adam/update_dense/bias/ApplyAdam	ApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/dense/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/bias*
use_nesterov( *
_output_shapes
:@
■
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/dense/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_1/kernel*
use_nesterov( *
_output_shapes

:@
ё
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/dense/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_1/bias*
use_nesterov( *
_output_shapes
:
╩
Adam/mulMulbeta1_power/read
Adam/beta1B^Adam/update_blstm_0/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdamD^Adam/update_blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdamB^Adam/update_blstm_0/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdamD^Adam/update_blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdamB^Adam/update_blstm_1/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdamD^Adam/update_blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdamB^Adam/update_blstm_1/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdamD^Adam/update_blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes
: 
╢
Adam/AssignAssignbeta1_powerAdam/mul*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
╠

Adam/mul_1Mulbeta2_power/read
Adam/beta2B^Adam/update_blstm_0/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdamD^Adam/update_blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdamB^Adam/update_blstm_0/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdamD^Adam/update_blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdamB^Adam/update_blstm_1/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdamD^Adam/update_blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdamB^Adam/update_blstm_1/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdamD^Adam/update_blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes
: 
║
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes
: 
ц
AdamNoOp^Adam/Assign^Adam/Assign_1B^Adam/update_blstm_0/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdamD^Adam/update_blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdamB^Adam/update_blstm_0/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdamD^Adam/update_blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdamB^Adam/update_blstm_1/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdamD^Adam/update_blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdamB^Adam/update_blstm_1/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdamD^Adam/update_blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam
[
accuracy/ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
Ш
accuracy/ArgMaxArgMaxdense/dense_1/Softmaxaccuracy/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:         *

Tidx0
]
accuracy/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
Ы
accuracy/ArgMax_1ArgMaxinputs/Placeholder_1accuracy/ArgMax_1/dimension*
output_type0	*#
_output_shapes
:         *

Tidx0*
T0
i
accuracy/EqualEqualaccuracy/ArgMaxaccuracy/ArgMax_1*#
_output_shapes
:         *
T0	
r
accuracy/CastCastaccuracy/Equal*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:         
X
accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
r
accuracy/MeanMeanaccuracy/Castaccuracy/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
R
loss_1/tagsConst*
valueB Bloss_1*
dtype0*
_output_shapes
: 
P
loss_1ScalarSummaryloss_1/tags	loss/Mean*
_output_shapes
: *
T0
Z
accuracy_1/tagsConst*
dtype0*
_output_shapes
: *
valueB B
accuracy_1
\

accuracy_1ScalarSummaryaccuracy_1/tagsaccuracy/Mean*
T0*
_output_shapes
: 
W
Merge/MergeSummaryMergeSummaryloss_1
accuracy_1*
N*
_output_shapes
: 
▐
initNoOp^beta1_power/Assign^beta2_power/Assign8^blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam/Assign:^blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Assign3^blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Assign:^blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Assign<^blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Assign5^blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Assign8^blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam/Assign:^blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Assign3^blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Assign:^blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Assign<^blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Assign5^blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Assign8^blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam/Assign:^blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Assign3^blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Assign:^blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Assign<^blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Assign5^blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Assign8^blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam/Assign:^blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Assign3^blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Assign:^blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Assign<^blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Assign5^blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^dense/bias/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign^dense_1/bias/Adam/Assign^dense_1/bias/Adam_1/Assign^dense_1/bias/Assign^dense_1/kernel/Adam/Assign^dense_1/kernel/Adam_1/Assign^dense_1/kernel/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_0986acebc3dc46bcaf3cca2e969735dc/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Д
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:&*╖
valueнBк&Bbeta1_powerBbeta2_powerB+blstm_0/bidirectional_rnn/bw/lstm_cell/biasB0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/AdamB2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B-blstm_0/bidirectional_rnn/bw/lstm_cell/kernelB2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/AdamB4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B+blstm_0/bidirectional_rnn/fw/lstm_cell/biasB0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/AdamB2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B-blstm_0/bidirectional_rnn/fw/lstm_cell/kernelB2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/AdamB4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B+blstm_1/bidirectional_rnn/bw/lstm_cell/biasB0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/AdamB2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B-blstm_1/bidirectional_rnn/bw/lstm_cell/kernelB2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/AdamB4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B+blstm_1/bidirectional_rnn/fw/lstm_cell/biasB0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/AdamB2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B-blstm_1/bidirectional_rnn/fw/lstm_cell/kernelB2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/AdamB4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1
п
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:&*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
╗
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_power+blstm_0/bidirectional_rnn/bw/lstm_cell/bias0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1-blstm_0/bidirectional_rnn/bw/lstm_cell/kernel2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1+blstm_0/bidirectional_rnn/fw/lstm_cell/bias0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1-blstm_0/bidirectional_rnn/fw/lstm_cell/kernel2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1+blstm_1/bidirectional_rnn/bw/lstm_cell/bias0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1-blstm_1/bidirectional_rnn/bw/lstm_cell/kernel2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1+blstm_1/bidirectional_rnn/fw/lstm_cell/bias0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1-blstm_1/bidirectional_rnn/fw/lstm_cell/kernel2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1dense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1dense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1*4
dtypes*
(2&
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
З
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:&*╖
valueнBк&Bbeta1_powerBbeta2_powerB+blstm_0/bidirectional_rnn/bw/lstm_cell/biasB0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/AdamB2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B-blstm_0/bidirectional_rnn/bw/lstm_cell/kernelB2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/AdamB4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B+blstm_0/bidirectional_rnn/fw/lstm_cell/biasB0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/AdamB2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B-blstm_0/bidirectional_rnn/fw/lstm_cell/kernelB2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/AdamB4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B+blstm_1/bidirectional_rnn/bw/lstm_cell/biasB0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/AdamB2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B-blstm_1/bidirectional_rnn/bw/lstm_cell/kernelB2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/AdamB4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B+blstm_1/bidirectional_rnn/fw/lstm_cell/biasB0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/AdamB2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B-blstm_1/bidirectional_rnn/fw/lstm_cell/kernelB2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/AdamB4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1
▓
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:&*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
╠
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*о
_output_shapesЫ
Ш::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&
╝
save/AssignAssignbeta1_powersave/RestoreV2*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes
: *
use_locking(
└
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes
: 
х
save/Assign_2Assign+blstm_0/bidirectional_rnn/bw/lstm_cell/biassave/RestoreV2:2*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ъ
save/Assign_3Assign0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adamsave/RestoreV2:3*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ь
save/Assign_4Assign2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save/RestoreV2:4*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias
ю
save/Assign_5Assign-blstm_0/bidirectional_rnn/bw/lstm_cell/kernelsave/RestoreV2:5*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА*
use_locking(
є
save/Assign_6Assign2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave/RestoreV2:6*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА*
use_locking(
ї
save/Assign_7Assign4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save/RestoreV2:7*
validate_shape(* 
_output_shapes
:
зА*
use_locking(*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel
х
save/Assign_8Assign+blstm_0/bidirectional_rnn/fw/lstm_cell/biassave/RestoreV2:8*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
ъ
save/Assign_9Assign0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adamsave/RestoreV2:9*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias
ю
save/Assign_10Assign2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save/RestoreV2:10*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias
Ё
save/Assign_11Assign-blstm_0/bidirectional_rnn/fw/lstm_cell/kernelsave/RestoreV2:11*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА*
use_locking(
ї
save/Assign_12Assign2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave/RestoreV2:12*
use_locking(*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА
ў
save/Assign_13Assign4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save/RestoreV2:13*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА*
use_locking(
ч
save/Assign_14Assign+blstm_1/bidirectional_rnn/bw/lstm_cell/biassave/RestoreV2:14*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias
ь
save/Assign_15Assign0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adamsave/RestoreV2:15*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ю
save/Assign_16Assign2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save/RestoreV2:16*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
Ё
save/Assign_17Assign-blstm_1/bidirectional_rnn/bw/lstm_cell/kernelsave/RestoreV2:17*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
ї
save/Assign_18Assign2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave/RestoreV2:18*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
ў
save/Assign_19Assign4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save/RestoreV2:19*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel
ч
save/Assign_20Assign+blstm_1/bidirectional_rnn/fw/lstm_cell/biassave/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ь
save/Assign_21Assign0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adamsave/RestoreV2:21*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
ю
save/Assign_22Assign2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save/RestoreV2:22*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias
Ё
save/Assign_23Assign-blstm_1/bidirectional_rnn/fw/lstm_cell/kernelsave/RestoreV2:23*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
ї
save/Assign_24Assign2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave/RestoreV2:24*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel
ў
save/Assign_25Assign4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save/RestoreV2:25*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
д
save/Assign_26Assign
dense/biassave/RestoreV2:26*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:@
й
save/Assign_27Assigndense/bias/Adamsave/RestoreV2:27*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:@
л
save/Assign_28Assigndense/bias/Adam_1save/RestoreV2:28*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
н
save/Assign_29Assigndense/kernelsave/RestoreV2:29*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	А@*
use_locking(
▓
save/Assign_30Assigndense/kernel/Adamsave/RestoreV2:30*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	А@
┤
save/Assign_31Assigndense/kernel/Adam_1save/RestoreV2:31*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	А@*
use_locking(
и
save/Assign_32Assigndense_1/biassave/RestoreV2:32*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
н
save/Assign_33Assigndense_1/bias/Adamsave/RestoreV2:33*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
п
save/Assign_34Assigndense_1/bias/Adam_1save/RestoreV2:34*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
░
save/Assign_35Assigndense_1/kernelsave/RestoreV2:35*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*!
_class
loc:@dense_1/kernel
╡
save/Assign_36Assigndense_1/kernel/Adamsave/RestoreV2:36*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*!
_class
loc:@dense_1/kernel
╖
save/Assign_37Assigndense_1/kernel/Adam_1save/RestoreV2:37*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:@
Ф
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
R
save/Const_1Const*
dtype0*
_output_shapes
: *
valueB Bmodel
Ж
save/StringJoin_1/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_20abb50f9d33412db1436be3e0a32958/part
{
save/StringJoin_1
StringJoinsave/Const_1save/StringJoin_1/inputs_1*
N*
_output_shapes
: *
	separator 
S
save/num_shards_1Const*
value	B :*
dtype0*
_output_shapes
: 
^
save/ShardedFilename_1/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save/ShardedFilename_1ShardedFilenamesave/StringJoin_1save/ShardedFilename_1/shardsave/num_shards_1*
_output_shapes
: 
Ж
save/SaveV2_1/tensor_namesConst*╖
valueнBк&Bbeta1_powerBbeta2_powerB+blstm_0/bidirectional_rnn/bw/lstm_cell/biasB0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/AdamB2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B-blstm_0/bidirectional_rnn/bw/lstm_cell/kernelB2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/AdamB4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B+blstm_0/bidirectional_rnn/fw/lstm_cell/biasB0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/AdamB2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B-blstm_0/bidirectional_rnn/fw/lstm_cell/kernelB2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/AdamB4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B+blstm_1/bidirectional_rnn/bw/lstm_cell/biasB0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/AdamB2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B-blstm_1/bidirectional_rnn/bw/lstm_cell/kernelB2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/AdamB4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B+blstm_1/bidirectional_rnn/fw/lstm_cell/biasB0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/AdamB2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B-blstm_1/bidirectional_rnn/fw/lstm_cell/kernelB2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/AdamB4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1*
dtype0*
_output_shapes
:&
▒
save/SaveV2_1/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
├
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicesbeta1_powerbeta2_power+blstm_0/bidirectional_rnn/bw/lstm_cell/bias0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1-blstm_0/bidirectional_rnn/bw/lstm_cell/kernel2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1+blstm_0/bidirectional_rnn/fw/lstm_cell/bias0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1-blstm_0/bidirectional_rnn/fw/lstm_cell/kernel2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1+blstm_1/bidirectional_rnn/bw/lstm_cell/bias0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1-blstm_1/bidirectional_rnn/bw/lstm_cell/kernel2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1+blstm_1/bidirectional_rnn/fw/lstm_cell/bias0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1-blstm_1/bidirectional_rnn/fw/lstm_cell/kernel2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1dense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1dense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1*4
dtypes*
(2&
Щ
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1*
T0*)
_class
loc:@save/ShardedFilename_1*
_output_shapes
: 
г
-save/MergeV2Checkpoints_1/checkpoint_prefixesPacksave/ShardedFilename_1^save/control_dependency_1*
N*
_output_shapes
:*
T0*

axis 
Г
save/MergeV2Checkpoints_1MergeV2Checkpoints-save/MergeV2Checkpoints_1/checkpoint_prefixessave/Const_1*
delete_old_dirs(
В
save/Identity_1Identitysave/Const_1^save/MergeV2Checkpoints_1^save/control_dependency_1*
_output_shapes
: *
T0
Й
save/RestoreV2_1/tensor_namesConst*
dtype0*
_output_shapes
:&*╖
valueнBк&Bbeta1_powerBbeta2_powerB+blstm_0/bidirectional_rnn/bw/lstm_cell/biasB0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/AdamB2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B-blstm_0/bidirectional_rnn/bw/lstm_cell/kernelB2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/AdamB4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B+blstm_0/bidirectional_rnn/fw/lstm_cell/biasB0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/AdamB2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B-blstm_0/bidirectional_rnn/fw/lstm_cell/kernelB2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/AdamB4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B+blstm_1/bidirectional_rnn/bw/lstm_cell/biasB0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/AdamB2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B-blstm_1/bidirectional_rnn/bw/lstm_cell/kernelB2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/AdamB4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B+blstm_1/bidirectional_rnn/fw/lstm_cell/biasB0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/AdamB2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B-blstm_1/bidirectional_rnn/fw/lstm_cell/kernelB2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/AdamB4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1
┤
!save/RestoreV2_1/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
╘
save/RestoreV2_1	RestoreV2save/Const_1save/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*о
_output_shapesЫ
Ш::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&
┴
save/Assign_38Assignbeta1_powersave/RestoreV2_1*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes
: *
use_locking(
├
save/Assign_39Assignbeta2_powersave/RestoreV2_1:1*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes
: 
ш
save/Assign_40Assign+blstm_0/bidirectional_rnn/bw/lstm_cell/biassave/RestoreV2_1:2*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias
э
save/Assign_41Assign0blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adamsave/RestoreV2_1:3*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias
я
save/Assign_42Assign2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save/RestoreV2_1:4*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ё
save/Assign_43Assign-blstm_0/bidirectional_rnn/bw/lstm_cell/kernelsave/RestoreV2_1:5*
use_locking(*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА
Ў
save/Assign_44Assign2blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave/RestoreV2_1:6*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА*
use_locking(
°
save/Assign_45Assign4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save/RestoreV2_1:7*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА*
use_locking(
ш
save/Assign_46Assign+blstm_0/bidirectional_rnn/fw/lstm_cell/biassave/RestoreV2_1:8*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
э
save/Assign_47Assign0blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adamsave/RestoreV2_1:9*
use_locking(*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
Ё
save/Assign_48Assign2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save/RestoreV2_1:10*
T0*>
_class4
20loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
Є
save/Assign_49Assign-blstm_0/bidirectional_rnn/fw/lstm_cell/kernelsave/RestoreV2_1:11*
use_locking(*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА
ў
save/Assign_50Assign2blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave/RestoreV2_1:12*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА*
use_locking(
∙
save/Assign_51Assign4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save/RestoreV2_1:13*
use_locking(*
T0*@
_class6
42loc:@blstm_0/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
зА
щ
save/Assign_52Assign+blstm_1/bidirectional_rnn/bw/lstm_cell/biassave/RestoreV2_1:14*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ю
save/Assign_53Assign0blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adamsave/RestoreV2_1:15*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias
Ё
save/Assign_54Assign2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save/RestoreV2_1:16*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/bias
Є
save/Assign_55Assign-blstm_1/bidirectional_rnn/bw/lstm_cell/kernelsave/RestoreV2_1:17*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
ў
save/Assign_56Assign2blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave/RestoreV2_1:18*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
∙
save/Assign_57Assign4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save/RestoreV2_1:19*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/bw/lstm_cell/kernel
щ
save/Assign_58Assign+blstm_1/bidirectional_rnn/fw/lstm_cell/biassave/RestoreV2_1:20*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
ю
save/Assign_59Assign0blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adamsave/RestoreV2_1:21*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
Ё
save/Assign_60Assign2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save/RestoreV2_1:22*
use_locking(*
T0*>
_class4
20loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
Є
save/Assign_61Assign-blstm_1/bidirectional_rnn/fw/lstm_cell/kernelsave/RestoreV2_1:23*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
ў
save/Assign_62Assign2blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave/RestoreV2_1:24*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
∙
save/Assign_63Assign4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save/RestoreV2_1:25*
use_locking(*
T0*@
_class6
42loc:@blstm_1/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
ж
save/Assign_64Assign
dense/biassave/RestoreV2_1:26*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:@
л
save/Assign_65Assigndense/bias/Adamsave/RestoreV2_1:27*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:@
н
save/Assign_66Assigndense/bias/Adam_1save/RestoreV2_1:28*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:@
п
save/Assign_67Assigndense/kernelsave/RestoreV2_1:29*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	А@
┤
save/Assign_68Assigndense/kernel/Adamsave/RestoreV2_1:30*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	А@
╢
save/Assign_69Assigndense/kernel/Adam_1save/RestoreV2_1:31*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	А@
к
save/Assign_70Assigndense_1/biassave/RestoreV2_1:32*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
п
save/Assign_71Assigndense_1/bias/Adamsave/RestoreV2_1:33*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
▒
save/Assign_72Assigndense_1/bias/Adam_1save/RestoreV2_1:34*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
▓
save/Assign_73Assigndense_1/kernelsave/RestoreV2_1:35*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
╖
save/Assign_74Assigndense_1/kernel/Adamsave/RestoreV2_1:36*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:@
╣
save/Assign_75Assigndense_1/kernel/Adam_1save/RestoreV2_1:37*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
в
save/restore_shard_1NoOp^save/Assign_38^save/Assign_39^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75
1
save/restore_all_1NoOp^save/restore_shard_1 "B
save/Const_1:0save/Identity_1:0save/restore_all_1 (5 @F8"┘
trainable_variables┴╛
ы
/blstm_0/bidirectional_rnn/fw/lstm_cell/kernel:04blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Assign4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/read:02Jblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform:08
┌
-blstm_0/bidirectional_rnn/fw/lstm_cell/bias:02blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Assign2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/read:02?blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros:08
ы
/blstm_0/bidirectional_rnn/bw/lstm_cell/kernel:04blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Assign4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/read:02Jblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform:08
┌
-blstm_0/bidirectional_rnn/bw/lstm_cell/bias:02blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Assign2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/read:02?blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros:08
ы
/blstm_1/bidirectional_rnn/fw/lstm_cell/kernel:04blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Assign4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/read:02Jblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform:08
┌
-blstm_1/bidirectional_rnn/fw/lstm_cell/bias:02blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Assign2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/read:02?blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros:08
ы
/blstm_1/bidirectional_rnn/bw/lstm_cell/kernel:04blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Assign4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/read:02Jblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform:08
┌
-blstm_1/bidirectional_rnn/bw/lstm_cell/bias:02blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Assign2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/read:02?blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08"'
	summaries

loss_1:0
accuracy_1:0"
train_op

Adam"│р
while_contextарЬр
╪М
3blstm_0/bidirectional_rnn/fw/fw/while/while_context *0blstm_0/bidirectional_rnn/fw/fw/while/LoopCond:02-blstm_0/bidirectional_rnn/fw/fw/while/Merge:0:0blstm_0/bidirectional_rnn/fw/fw/while/Identity:0B,blstm_0/bidirectional_rnn/fw/fw/while/Exit:0B.blstm_0/bidirectional_rnn/fw/fw/while/Exit_1:0B.blstm_0/bidirectional_rnn/fw/fw/while/Exit_2:0B.blstm_0/bidirectional_rnn/fw/fw/while/Exit_3:0B.blstm_0/bidirectional_rnn/fw/fw/while/Exit_4:0Bgradients/f_count_8:0J╞Ж
-blstm_0/bidirectional_rnn/fw/fw/CheckSeqLen:0
)blstm_0/bidirectional_rnn/fw/fw/Minimum:0
-blstm_0/bidirectional_rnn/fw/fw/TensorArray:0
\blstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
/blstm_0/bidirectional_rnn/fw/fw/TensorArray_1:0
1blstm_0/bidirectional_rnn/fw/fw/strided_slice_1:0
-blstm_0/bidirectional_rnn/fw/fw/while/Enter:0
/blstm_0/bidirectional_rnn/fw/fw/while/Enter_1:0
/blstm_0/bidirectional_rnn/fw/fw/while/Enter_2:0
/blstm_0/bidirectional_rnn/fw/fw/while/Enter_3:0
/blstm_0/bidirectional_rnn/fw/fw/while/Enter_4:0
,blstm_0/bidirectional_rnn/fw/fw/while/Exit:0
.blstm_0/bidirectional_rnn/fw/fw/while/Exit_1:0
.blstm_0/bidirectional_rnn/fw/fw/while/Exit_2:0
.blstm_0/bidirectional_rnn/fw/fw/while/Exit_3:0
.blstm_0/bidirectional_rnn/fw/fw/while/Exit_4:0
:blstm_0/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0
4blstm_0/bidirectional_rnn/fw/fw/while/GreaterEqual:0
0blstm_0/bidirectional_rnn/fw/fw/while/Identity:0
2blstm_0/bidirectional_rnn/fw/fw/while/Identity_1:0
2blstm_0/bidirectional_rnn/fw/fw/while/Identity_2:0
2blstm_0/bidirectional_rnn/fw/fw/while/Identity_3:0
2blstm_0/bidirectional_rnn/fw/fw/while/Identity_4:0
2blstm_0/bidirectional_rnn/fw/fw/while/Less/Enter:0
,blstm_0/bidirectional_rnn/fw/fw/while/Less:0
4blstm_0/bidirectional_rnn/fw/fw/while/Less_1/Enter:0
.blstm_0/bidirectional_rnn/fw/fw/while/Less_1:0
2blstm_0/bidirectional_rnn/fw/fw/while/LogicalAnd:0
0blstm_0/bidirectional_rnn/fw/fw/while/LoopCond:0
-blstm_0/bidirectional_rnn/fw/fw/while/Merge:0
-blstm_0/bidirectional_rnn/fw/fw/while/Merge:1
/blstm_0/bidirectional_rnn/fw/fw/while/Merge_1:0
/blstm_0/bidirectional_rnn/fw/fw/while/Merge_1:1
/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2:0
/blstm_0/bidirectional_rnn/fw/fw/while/Merge_2:1
/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3:0
/blstm_0/bidirectional_rnn/fw/fw/while/Merge_3:1
/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4:0
/blstm_0/bidirectional_rnn/fw/fw/while/Merge_4:1
5blstm_0/bidirectional_rnn/fw/fw/while/NextIteration:0
7blstm_0/bidirectional_rnn/fw/fw/while/NextIteration_1:0
7blstm_0/bidirectional_rnn/fw/fw/while/NextIteration_2:0
7blstm_0/bidirectional_rnn/fw/fw/while/NextIteration_3:0
7blstm_0/bidirectional_rnn/fw/fw/while/NextIteration_4:0
4blstm_0/bidirectional_rnn/fw/fw/while/Select/Enter:0
.blstm_0/bidirectional_rnn/fw/fw/while/Select:0
0blstm_0/bidirectional_rnn/fw/fw/while/Select_1:0
0blstm_0/bidirectional_rnn/fw/fw/while/Select_2:0
.blstm_0/bidirectional_rnn/fw/fw/while/Switch:0
.blstm_0/bidirectional_rnn/fw/fw/while/Switch:1
0blstm_0/bidirectional_rnn/fw/fw/while/Switch_1:0
0blstm_0/bidirectional_rnn/fw/fw/while/Switch_1:1
0blstm_0/bidirectional_rnn/fw/fw/while/Switch_2:0
0blstm_0/bidirectional_rnn/fw/fw/while/Switch_2:1
0blstm_0/bidirectional_rnn/fw/fw/while/Switch_3:0
0blstm_0/bidirectional_rnn/fw/fw/while/Switch_3:1
0blstm_0/bidirectional_rnn/fw/fw/while/Switch_4:0
0blstm_0/bidirectional_rnn/fw/fw/while/Switch_4:1
?blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0
Ablstm_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0
9blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3:0
Qblstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Kblstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3:0
-blstm_0/bidirectional_rnn/fw/fw/while/add/y:0
+blstm_0/bidirectional_rnn/fw/fw/while/add:0
/blstm_0/bidirectional_rnn/fw/fw/while/add_1/y:0
-blstm_0/bidirectional_rnn/fw/fw/while/add_1:0
5blstm_0/bidirectional_rnn/fw/fw/while/dropout/Floor:0
5blstm_0/bidirectional_rnn/fw/fw/while/dropout/Shape:0
3blstm_0/bidirectional_rnn/fw/fw/while/dropout/add:0
3blstm_0/bidirectional_rnn/fw/fw/while/dropout/div:0
9blstm_0/bidirectional_rnn/fw/fw/while/dropout/keep_prob:0
3blstm_0/bidirectional_rnn/fw/fw/while/dropout/mul:0
Lblstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniform:0
Bblstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/max:0
Bblstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/min:0
Bblstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/mul:0
Bblstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform/sub:0
>blstm_0/bidirectional_rnn/fw/fw/while/dropout/random_uniform:0
?blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter:0
9blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd:0
7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Const:0
>blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter:0
8blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul:0
9blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid:0
;blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1:0
;blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2:0
6blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh:0
8blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1:0
7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add/y:0
5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add:0
7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1:0
=blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat/axis:0
8blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat:0
5blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul:0
7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1:0
7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2:0
Ablstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split/split_dim:0
7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split:0
7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split:1
7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split:2
7blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/split:3
'blstm_0/bidirectional_rnn/fw/fw/zeros:0
2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/read:0
4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/read:0
gradients/Add_2/y:0
gradients/Add_2:0
gradients/Merge_4:0
gradients/Merge_4:1
gradients/NextIteration_4:0
gradients/Switch_4:0
gradients/Switch_4:1
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter:0
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2:0
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc:0
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Enter:0
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPushV2:0
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc:0
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Enter:0
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPushV2:0
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc:0
rgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
xgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
rgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0
\gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape:0
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1:0
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
fgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape:0
Mgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Shape:0
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Enter:0
Zgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPushV2:0
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc:0
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Enter:0
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc:0
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
Xgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape:0
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1:0
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Enter:0
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc:0
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0
Xgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape:0
Ngradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1:0
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
fgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
hgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Enter:0
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc:0
Jgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape:0
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1:0
gradients/f_count_6:0
gradients/f_count_7:0
gradients/f_count_8:0g
1blstm_0/bidirectional_rnn/fw/fw/strided_slice_1:02blstm_0/bidirectional_rnn/fw/fw/while/Less/Enter:0╠
dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0u
2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/read:0?blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter:0╚
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0─
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0б
\blstm_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Ablstm_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0и
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0и
Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0Rgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0д
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc:0Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Enter:0Ь
Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc:0Lgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter:0ш
rgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0rgradients/blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0░
Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0Vgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0д
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc:0Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Enter:0д
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc:0Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Enter:0В
-blstm_0/bidirectional_rnn/fw/fw/TensorArray:0Qblstm_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0╚
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0╚
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0д
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc:0Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Enter:0v
4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/read:0>blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter:0╠
dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0м
Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc:0Tgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Enter:0╚
bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0bgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0_
'blstm_0/bidirectional_rnn/fw/fw/zeros:04blstm_0/bidirectional_rnn/fw/fw/while/Select/Enter:0─
`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0`gradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0a
)blstm_0/bidirectional_rnn/fw/fw/Minimum:04blstm_0/bidirectional_rnn/fw/fw/while/Less_1/Enter:0╠
dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0dgradients/blstm_0/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0r
/blstm_0/bidirectional_rnn/fw/fw/TensorArray_1:0?blstm_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0д
Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc:0Pgradients/blstm_0/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Enter:0k
-blstm_0/bidirectional_rnn/fw/fw/CheckSeqLen:0:blstm_0/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0R-blstm_0/bidirectional_rnn/fw/fw/while/Enter:0R/blstm_0/bidirectional_rnn/fw/fw/while/Enter_1:0R/blstm_0/bidirectional_rnn/fw/fw/while/Enter_2:0R/blstm_0/bidirectional_rnn/fw/fw/while/Enter_3:0R/blstm_0/bidirectional_rnn/fw/fw/while/Enter_4:0Rgradients/f_count_7:0Z1blstm_0/bidirectional_rnn/fw/fw/strided_slice_1:0
▄М
3blstm_0/bidirectional_rnn/bw/bw/while/while_context *0blstm_0/bidirectional_rnn/bw/bw/while/LoopCond:02-blstm_0/bidirectional_rnn/bw/bw/while/Merge:0:0blstm_0/bidirectional_rnn/bw/bw/while/Identity:0B,blstm_0/bidirectional_rnn/bw/bw/while/Exit:0B.blstm_0/bidirectional_rnn/bw/bw/while/Exit_1:0B.blstm_0/bidirectional_rnn/bw/bw/while/Exit_2:0B.blstm_0/bidirectional_rnn/bw/bw/while/Exit_3:0B.blstm_0/bidirectional_rnn/bw/bw/while/Exit_4:0Bgradients/f_count_11:0J╚Ж
-blstm_0/bidirectional_rnn/bw/bw/CheckSeqLen:0
)blstm_0/bidirectional_rnn/bw/bw/Minimum:0
-blstm_0/bidirectional_rnn/bw/bw/TensorArray:0
\blstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
/blstm_0/bidirectional_rnn/bw/bw/TensorArray_1:0
1blstm_0/bidirectional_rnn/bw/bw/strided_slice_1:0
-blstm_0/bidirectional_rnn/bw/bw/while/Enter:0
/blstm_0/bidirectional_rnn/bw/bw/while/Enter_1:0
/blstm_0/bidirectional_rnn/bw/bw/while/Enter_2:0
/blstm_0/bidirectional_rnn/bw/bw/while/Enter_3:0
/blstm_0/bidirectional_rnn/bw/bw/while/Enter_4:0
,blstm_0/bidirectional_rnn/bw/bw/while/Exit:0
.blstm_0/bidirectional_rnn/bw/bw/while/Exit_1:0
.blstm_0/bidirectional_rnn/bw/bw/while/Exit_2:0
.blstm_0/bidirectional_rnn/bw/bw/while/Exit_3:0
.blstm_0/bidirectional_rnn/bw/bw/while/Exit_4:0
:blstm_0/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0
4blstm_0/bidirectional_rnn/bw/bw/while/GreaterEqual:0
0blstm_0/bidirectional_rnn/bw/bw/while/Identity:0
2blstm_0/bidirectional_rnn/bw/bw/while/Identity_1:0
2blstm_0/bidirectional_rnn/bw/bw/while/Identity_2:0
2blstm_0/bidirectional_rnn/bw/bw/while/Identity_3:0
2blstm_0/bidirectional_rnn/bw/bw/while/Identity_4:0
2blstm_0/bidirectional_rnn/bw/bw/while/Less/Enter:0
,blstm_0/bidirectional_rnn/bw/bw/while/Less:0
4blstm_0/bidirectional_rnn/bw/bw/while/Less_1/Enter:0
.blstm_0/bidirectional_rnn/bw/bw/while/Less_1:0
2blstm_0/bidirectional_rnn/bw/bw/while/LogicalAnd:0
0blstm_0/bidirectional_rnn/bw/bw/while/LoopCond:0
-blstm_0/bidirectional_rnn/bw/bw/while/Merge:0
-blstm_0/bidirectional_rnn/bw/bw/while/Merge:1
/blstm_0/bidirectional_rnn/bw/bw/while/Merge_1:0
/blstm_0/bidirectional_rnn/bw/bw/while/Merge_1:1
/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2:0
/blstm_0/bidirectional_rnn/bw/bw/while/Merge_2:1
/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3:0
/blstm_0/bidirectional_rnn/bw/bw/while/Merge_3:1
/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4:0
/blstm_0/bidirectional_rnn/bw/bw/while/Merge_4:1
5blstm_0/bidirectional_rnn/bw/bw/while/NextIteration:0
7blstm_0/bidirectional_rnn/bw/bw/while/NextIteration_1:0
7blstm_0/bidirectional_rnn/bw/bw/while/NextIteration_2:0
7blstm_0/bidirectional_rnn/bw/bw/while/NextIteration_3:0
7blstm_0/bidirectional_rnn/bw/bw/while/NextIteration_4:0
4blstm_0/bidirectional_rnn/bw/bw/while/Select/Enter:0
.blstm_0/bidirectional_rnn/bw/bw/while/Select:0
0blstm_0/bidirectional_rnn/bw/bw/while/Select_1:0
0blstm_0/bidirectional_rnn/bw/bw/while/Select_2:0
.blstm_0/bidirectional_rnn/bw/bw/while/Switch:0
.blstm_0/bidirectional_rnn/bw/bw/while/Switch:1
0blstm_0/bidirectional_rnn/bw/bw/while/Switch_1:0
0blstm_0/bidirectional_rnn/bw/bw/while/Switch_1:1
0blstm_0/bidirectional_rnn/bw/bw/while/Switch_2:0
0blstm_0/bidirectional_rnn/bw/bw/while/Switch_2:1
0blstm_0/bidirectional_rnn/bw/bw/while/Switch_3:0
0blstm_0/bidirectional_rnn/bw/bw/while/Switch_3:1
0blstm_0/bidirectional_rnn/bw/bw/while/Switch_4:0
0blstm_0/bidirectional_rnn/bw/bw/while/Switch_4:1
?blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0
Ablstm_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0
9blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3:0
Qblstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Kblstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3:0
-blstm_0/bidirectional_rnn/bw/bw/while/add/y:0
+blstm_0/bidirectional_rnn/bw/bw/while/add:0
/blstm_0/bidirectional_rnn/bw/bw/while/add_1/y:0
-blstm_0/bidirectional_rnn/bw/bw/while/add_1:0
5blstm_0/bidirectional_rnn/bw/bw/while/dropout/Floor:0
5blstm_0/bidirectional_rnn/bw/bw/while/dropout/Shape:0
3blstm_0/bidirectional_rnn/bw/bw/while/dropout/add:0
3blstm_0/bidirectional_rnn/bw/bw/while/dropout/div:0
9blstm_0/bidirectional_rnn/bw/bw/while/dropout/keep_prob:0
3blstm_0/bidirectional_rnn/bw/bw/while/dropout/mul:0
Lblstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniform:0
Bblstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/max:0
Bblstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/min:0
Bblstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/mul:0
Bblstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform/sub:0
>blstm_0/bidirectional_rnn/bw/bw/while/dropout/random_uniform:0
?blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter:0
9blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd:0
7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Const:0
>blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter:0
8blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul:0
9blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid:0
;blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1:0
;blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2:0
6blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh:0
8blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1:0
7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add/y:0
5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add:0
7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1:0
=blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat/axis:0
8blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat:0
5blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul:0
7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1:0
7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2:0
Ablstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split/split_dim:0
7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split:0
7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split:1
7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split:2
7blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/split:3
'blstm_0/bidirectional_rnn/bw/bw/zeros:0
2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/read:0
4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/read:0
gradients/Add_3/y:0
gradients/Add_3:0
gradients/Merge_6:0
gradients/Merge_6:1
gradients/NextIteration_6:0
gradients/Switch_6:0
gradients/Switch_6:1
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter:0
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2:0
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc:0
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Enter:0
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPushV2:0
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc:0
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Enter:0
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPushV2:0
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc:0
rgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
xgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
rgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0
\gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape:0
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1:0
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
fgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape:0
Mgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Shape:0
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Enter:0
Zgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPushV2:0
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc:0
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Enter:0
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc:0
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
Xgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape:0
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1:0
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Enter:0
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc:0
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0
Xgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape:0
Ngradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1:0
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
fgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
hgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Enter:0
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc:0
Jgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape:0
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1:0
gradients/f_count_10:0
gradients/f_count_11:0
gradients/f_count_9:0╠
dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0В
-blstm_0/bidirectional_rnn/bw/bw/TensorArray:0Qblstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0╚
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0u
2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/read:0?blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter:0─
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0б
\blstm_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Ablstm_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0v
4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/read:0>blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter:0_
'blstm_0/bidirectional_rnn/bw/bw/zeros:04blstm_0/bidirectional_rnn/bw/bw/while/Select/Enter:0и
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0a
)blstm_0/bidirectional_rnn/bw/bw/Minimum:04blstm_0/bidirectional_rnn/bw/bw/while/Less_1/Enter:0и
Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0Rgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0r
/blstm_0/bidirectional_rnn/bw/bw/TensorArray_1:0?blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0Ь
Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc:0Lgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter:0д
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc:0Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Enter:0ш
rgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0rgradients/blstm_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0k
-blstm_0/bidirectional_rnn/bw/bw/CheckSeqLen:0:blstm_0/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0░
Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0Vgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0g
1blstm_0/bidirectional_rnn/bw/bw/strided_slice_1:02blstm_0/bidirectional_rnn/bw/bw/while/Less/Enter:0д
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc:0Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Enter:0д
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc:0Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Enter:0╚
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0╚
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0д
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc:0Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Enter:0╠
dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0м
Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc:0Tgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Enter:0╚
bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0bgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0─
`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0`gradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0╠
dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0dgradients/blstm_0/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0д
Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc:0Pgradients/blstm_0/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Enter:0R-blstm_0/bidirectional_rnn/bw/bw/while/Enter:0R/blstm_0/bidirectional_rnn/bw/bw/while/Enter_1:0R/blstm_0/bidirectional_rnn/bw/bw/while/Enter_2:0R/blstm_0/bidirectional_rnn/bw/bw/while/Enter_3:0R/blstm_0/bidirectional_rnn/bw/bw/while/Enter_4:0Rgradients/f_count_10:0Z1blstm_0/bidirectional_rnn/bw/bw/strided_slice_1:0
дг
3blstm_1/bidirectional_rnn/fw/fw/while/while_context *0blstm_1/bidirectional_rnn/fw/fw/while/LoopCond:02-blstm_1/bidirectional_rnn/fw/fw/while/Merge:0:0blstm_1/bidirectional_rnn/fw/fw/while/Identity:0B,blstm_1/bidirectional_rnn/fw/fw/while/Exit:0B.blstm_1/bidirectional_rnn/fw/fw/while/Exit_1:0B.blstm_1/bidirectional_rnn/fw/fw/while/Exit_2:0B.blstm_1/bidirectional_rnn/fw/fw/while/Exit_3:0B.blstm_1/bidirectional_rnn/fw/fw/while/Exit_4:0Bgradients/f_count_2:0JТЭ
-blstm_1/bidirectional_rnn/fw/fw/CheckSeqLen:0
)blstm_1/bidirectional_rnn/fw/fw/Minimum:0
-blstm_1/bidirectional_rnn/fw/fw/TensorArray:0
\blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
/blstm_1/bidirectional_rnn/fw/fw/TensorArray_1:0
1blstm_1/bidirectional_rnn/fw/fw/strided_slice_1:0
-blstm_1/bidirectional_rnn/fw/fw/while/Enter:0
/blstm_1/bidirectional_rnn/fw/fw/while/Enter_1:0
/blstm_1/bidirectional_rnn/fw/fw/while/Enter_2:0
/blstm_1/bidirectional_rnn/fw/fw/while/Enter_3:0
/blstm_1/bidirectional_rnn/fw/fw/while/Enter_4:0
,blstm_1/bidirectional_rnn/fw/fw/while/Exit:0
.blstm_1/bidirectional_rnn/fw/fw/while/Exit_1:0
.blstm_1/bidirectional_rnn/fw/fw/while/Exit_2:0
.blstm_1/bidirectional_rnn/fw/fw/while/Exit_3:0
.blstm_1/bidirectional_rnn/fw/fw/while/Exit_4:0
:blstm_1/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0
4blstm_1/bidirectional_rnn/fw/fw/while/GreaterEqual:0
0blstm_1/bidirectional_rnn/fw/fw/while/Identity:0
2blstm_1/bidirectional_rnn/fw/fw/while/Identity_1:0
2blstm_1/bidirectional_rnn/fw/fw/while/Identity_2:0
2blstm_1/bidirectional_rnn/fw/fw/while/Identity_3:0
2blstm_1/bidirectional_rnn/fw/fw/while/Identity_4:0
2blstm_1/bidirectional_rnn/fw/fw/while/Less/Enter:0
,blstm_1/bidirectional_rnn/fw/fw/while/Less:0
4blstm_1/bidirectional_rnn/fw/fw/while/Less_1/Enter:0
.blstm_1/bidirectional_rnn/fw/fw/while/Less_1:0
2blstm_1/bidirectional_rnn/fw/fw/while/LogicalAnd:0
0blstm_1/bidirectional_rnn/fw/fw/while/LoopCond:0
-blstm_1/bidirectional_rnn/fw/fw/while/Merge:0
-blstm_1/bidirectional_rnn/fw/fw/while/Merge:1
/blstm_1/bidirectional_rnn/fw/fw/while/Merge_1:0
/blstm_1/bidirectional_rnn/fw/fw/while/Merge_1:1
/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2:0
/blstm_1/bidirectional_rnn/fw/fw/while/Merge_2:1
/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3:0
/blstm_1/bidirectional_rnn/fw/fw/while/Merge_3:1
/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4:0
/blstm_1/bidirectional_rnn/fw/fw/while/Merge_4:1
5blstm_1/bidirectional_rnn/fw/fw/while/NextIteration:0
7blstm_1/bidirectional_rnn/fw/fw/while/NextIteration_1:0
7blstm_1/bidirectional_rnn/fw/fw/while/NextIteration_2:0
7blstm_1/bidirectional_rnn/fw/fw/while/NextIteration_3:0
7blstm_1/bidirectional_rnn/fw/fw/while/NextIteration_4:0
4blstm_1/bidirectional_rnn/fw/fw/while/Select/Enter:0
.blstm_1/bidirectional_rnn/fw/fw/while/Select:0
0blstm_1/bidirectional_rnn/fw/fw/while/Select_1:0
0blstm_1/bidirectional_rnn/fw/fw/while/Select_2:0
.blstm_1/bidirectional_rnn/fw/fw/while/Switch:0
.blstm_1/bidirectional_rnn/fw/fw/while/Switch:1
0blstm_1/bidirectional_rnn/fw/fw/while/Switch_1:0
0blstm_1/bidirectional_rnn/fw/fw/while/Switch_1:1
0blstm_1/bidirectional_rnn/fw/fw/while/Switch_2:0
0blstm_1/bidirectional_rnn/fw/fw/while/Switch_2:1
0blstm_1/bidirectional_rnn/fw/fw/while/Switch_3:0
0blstm_1/bidirectional_rnn/fw/fw/while/Switch_3:1
0blstm_1/bidirectional_rnn/fw/fw/while/Switch_4:0
0blstm_1/bidirectional_rnn/fw/fw/while/Switch_4:1
?blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0
Ablstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0
9blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3:0
Qblstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Kblstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3:0
-blstm_1/bidirectional_rnn/fw/fw/while/add/y:0
+blstm_1/bidirectional_rnn/fw/fw/while/add:0
/blstm_1/bidirectional_rnn/fw/fw/while/add_1/y:0
-blstm_1/bidirectional_rnn/fw/fw/while/add_1:0
5blstm_1/bidirectional_rnn/fw/fw/while/dropout/Floor:0
5blstm_1/bidirectional_rnn/fw/fw/while/dropout/Shape:0
3blstm_1/bidirectional_rnn/fw/fw/while/dropout/add:0
3blstm_1/bidirectional_rnn/fw/fw/while/dropout/div:0
9blstm_1/bidirectional_rnn/fw/fw/while/dropout/keep_prob:0
3blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul:0
Lblstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniform:0
Bblstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/max:0
Bblstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/min:0
Bblstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/mul:0
Bblstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform/sub:0
>blstm_1/bidirectional_rnn/fw/fw/while/dropout/random_uniform:0
?blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter:0
9blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd:0
7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Const:0
>blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter:0
8blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul:0
9blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid:0
;blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1:0
;blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2:0
6blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh:0
8blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1:0
7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add/y:0
5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add:0
7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1:0
=blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat/axis:0
8blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat:0
5blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul:0
7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1:0
7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2:0
Ablstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split/split_dim:0
7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split:0
7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split:1
7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split:2
7blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/split:3
'blstm_1/bidirectional_rnn/fw/fw/zeros:0
2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/read:0
4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/read:0
gradients/Add/y:0
gradients/Add:0
gradients/Merge:0
gradients/Merge:1
gradients/NextIteration:0
gradients/Switch:0
gradients/Switch:1
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter:0
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2:0
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc:0
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Enter:0
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPushV2:0
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc:0
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Enter:0
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPushV2:0
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc:0
rgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
xgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
rgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc:0
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/Enter:0
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPushV2:0
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/f_acc:0
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape:0
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Enter:0
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2:0
fgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc:0
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Enter:0
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPushV2:0
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc:0
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Enter:0
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPushV2:0
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc:0
Hgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape:0
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1:0
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0
\gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape:0
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1:0
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
fgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape:0
Mgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Shape:0
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Enter:0
Zgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPushV2:0
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc:0
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Enter:0
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc:0
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
Xgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape:0
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1:0
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Enter:0
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc:0
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0
Xgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape:0
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1:0
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
fgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
hgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Enter:0
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc:0
Jgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape:0
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1:0
gradients/f_count:0
gradients/f_count_1:0
gradients/f_count_2:0д
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc:0Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Enter:0д
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc:0Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Enter:0_
'blstm_1/bidirectional_rnn/fw/fw/zeros:04blstm_1/bidirectional_rnn/fw/fw/while/Select/Enter:0╚
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0╠
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0д
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc:0Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Enter:0а
Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc:0Ngradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Enter:0Ь
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc:0Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter:0g
1blstm_1/bidirectional_rnn/fw/fw/strided_slice_1:02blstm_1/bidirectional_rnn/fw/fw/while/Less/Enter:0└
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc:0^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/Enter:0Ь
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/f_acc:0Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/Enter:0─
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0╠
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0v
4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/read:0>blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter:0д
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc:0Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Enter:0б
\blstm_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Ablstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0k
-blstm_1/bidirectional_rnn/fw/fw/CheckSeqLen:0:blstm_1/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0└
^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc:0^gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Enter:0─
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0`gradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0╚
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0a
)blstm_1/bidirectional_rnn/fw/fw/Minimum:04blstm_1/bidirectional_rnn/fw/fw/while/Less_1/Enter:0░
Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0Vgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0и
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0r
/blstm_1/bidirectional_rnn/fw/fw/TensorArray_1:0?blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0Ь
Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc:0Lgradients/blstm_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Enter:0─
`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0`gradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0и
Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0Rgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0ш
rgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0rgradients/blstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0u
2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/read:0?blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter:0д
Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc:0Pgradients/blstm_1/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Enter:0╚
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0╠
dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0dgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0м
Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc:0Tgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Enter:0╚
bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0bgradients/blstm_1/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0В
-blstm_1/bidirectional_rnn/fw/fw/TensorArray:0Qblstm_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0R-blstm_1/bidirectional_rnn/fw/fw/while/Enter:0R/blstm_1/bidirectional_rnn/fw/fw/while/Enter_1:0R/blstm_1/bidirectional_rnn/fw/fw/while/Enter_2:0R/blstm_1/bidirectional_rnn/fw/fw/while/Enter_3:0R/blstm_1/bidirectional_rnn/fw/fw/while/Enter_4:0Rgradients/f_count_1:0Z1blstm_1/bidirectional_rnn/fw/fw/strided_slice_1:0
┤г
3blstm_1/bidirectional_rnn/bw/bw/while/while_context *0blstm_1/bidirectional_rnn/bw/bw/while/LoopCond:02-blstm_1/bidirectional_rnn/bw/bw/while/Merge:0:0blstm_1/bidirectional_rnn/bw/bw/while/Identity:0B,blstm_1/bidirectional_rnn/bw/bw/while/Exit:0B.blstm_1/bidirectional_rnn/bw/bw/while/Exit_1:0B.blstm_1/bidirectional_rnn/bw/bw/while/Exit_2:0B.blstm_1/bidirectional_rnn/bw/bw/while/Exit_3:0B.blstm_1/bidirectional_rnn/bw/bw/while/Exit_4:0Bgradients/f_count_5:0JвЭ
-blstm_1/bidirectional_rnn/bw/bw/CheckSeqLen:0
)blstm_1/bidirectional_rnn/bw/bw/Minimum:0
-blstm_1/bidirectional_rnn/bw/bw/TensorArray:0
\blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
/blstm_1/bidirectional_rnn/bw/bw/TensorArray_1:0
1blstm_1/bidirectional_rnn/bw/bw/strided_slice_1:0
-blstm_1/bidirectional_rnn/bw/bw/while/Enter:0
/blstm_1/bidirectional_rnn/bw/bw/while/Enter_1:0
/blstm_1/bidirectional_rnn/bw/bw/while/Enter_2:0
/blstm_1/bidirectional_rnn/bw/bw/while/Enter_3:0
/blstm_1/bidirectional_rnn/bw/bw/while/Enter_4:0
,blstm_1/bidirectional_rnn/bw/bw/while/Exit:0
.blstm_1/bidirectional_rnn/bw/bw/while/Exit_1:0
.blstm_1/bidirectional_rnn/bw/bw/while/Exit_2:0
.blstm_1/bidirectional_rnn/bw/bw/while/Exit_3:0
.blstm_1/bidirectional_rnn/bw/bw/while/Exit_4:0
:blstm_1/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0
4blstm_1/bidirectional_rnn/bw/bw/while/GreaterEqual:0
0blstm_1/bidirectional_rnn/bw/bw/while/Identity:0
2blstm_1/bidirectional_rnn/bw/bw/while/Identity_1:0
2blstm_1/bidirectional_rnn/bw/bw/while/Identity_2:0
2blstm_1/bidirectional_rnn/bw/bw/while/Identity_3:0
2blstm_1/bidirectional_rnn/bw/bw/while/Identity_4:0
2blstm_1/bidirectional_rnn/bw/bw/while/Less/Enter:0
,blstm_1/bidirectional_rnn/bw/bw/while/Less:0
4blstm_1/bidirectional_rnn/bw/bw/while/Less_1/Enter:0
.blstm_1/bidirectional_rnn/bw/bw/while/Less_1:0
2blstm_1/bidirectional_rnn/bw/bw/while/LogicalAnd:0
0blstm_1/bidirectional_rnn/bw/bw/while/LoopCond:0
-blstm_1/bidirectional_rnn/bw/bw/while/Merge:0
-blstm_1/bidirectional_rnn/bw/bw/while/Merge:1
/blstm_1/bidirectional_rnn/bw/bw/while/Merge_1:0
/blstm_1/bidirectional_rnn/bw/bw/while/Merge_1:1
/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2:0
/blstm_1/bidirectional_rnn/bw/bw/while/Merge_2:1
/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3:0
/blstm_1/bidirectional_rnn/bw/bw/while/Merge_3:1
/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4:0
/blstm_1/bidirectional_rnn/bw/bw/while/Merge_4:1
5blstm_1/bidirectional_rnn/bw/bw/while/NextIteration:0
7blstm_1/bidirectional_rnn/bw/bw/while/NextIteration_1:0
7blstm_1/bidirectional_rnn/bw/bw/while/NextIteration_2:0
7blstm_1/bidirectional_rnn/bw/bw/while/NextIteration_3:0
7blstm_1/bidirectional_rnn/bw/bw/while/NextIteration_4:0
4blstm_1/bidirectional_rnn/bw/bw/while/Select/Enter:0
.blstm_1/bidirectional_rnn/bw/bw/while/Select:0
0blstm_1/bidirectional_rnn/bw/bw/while/Select_1:0
0blstm_1/bidirectional_rnn/bw/bw/while/Select_2:0
.blstm_1/bidirectional_rnn/bw/bw/while/Switch:0
.blstm_1/bidirectional_rnn/bw/bw/while/Switch:1
0blstm_1/bidirectional_rnn/bw/bw/while/Switch_1:0
0blstm_1/bidirectional_rnn/bw/bw/while/Switch_1:1
0blstm_1/bidirectional_rnn/bw/bw/while/Switch_2:0
0blstm_1/bidirectional_rnn/bw/bw/while/Switch_2:1
0blstm_1/bidirectional_rnn/bw/bw/while/Switch_3:0
0blstm_1/bidirectional_rnn/bw/bw/while/Switch_3:1
0blstm_1/bidirectional_rnn/bw/bw/while/Switch_4:0
0blstm_1/bidirectional_rnn/bw/bw/while/Switch_4:1
?blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0
Ablstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0
9blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3:0
Qblstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Kblstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3:0
-blstm_1/bidirectional_rnn/bw/bw/while/add/y:0
+blstm_1/bidirectional_rnn/bw/bw/while/add:0
/blstm_1/bidirectional_rnn/bw/bw/while/add_1/y:0
-blstm_1/bidirectional_rnn/bw/bw/while/add_1:0
5blstm_1/bidirectional_rnn/bw/bw/while/dropout/Floor:0
5blstm_1/bidirectional_rnn/bw/bw/while/dropout/Shape:0
3blstm_1/bidirectional_rnn/bw/bw/while/dropout/add:0
3blstm_1/bidirectional_rnn/bw/bw/while/dropout/div:0
9blstm_1/bidirectional_rnn/bw/bw/while/dropout/keep_prob:0
3blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul:0
Lblstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniform:0
Bblstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/max:0
Bblstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/min:0
Bblstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/mul:0
Bblstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform/sub:0
>blstm_1/bidirectional_rnn/bw/bw/while/dropout/random_uniform:0
?blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter:0
9blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd:0
7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Const:0
>blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter:0
8blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul:0
9blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid:0
;blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1:0
;blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2:0
6blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh:0
8blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1:0
7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add/y:0
5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add:0
7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1:0
=blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat/axis:0
8blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat:0
5blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul:0
7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1:0
7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2:0
Ablstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split/split_dim:0
7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split:0
7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split:1
7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split:2
7blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/split:3
'blstm_1/bidirectional_rnn/bw/bw/zeros:0
2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/read:0
4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/read:0
gradients/Add_1/y:0
gradients/Add_1:0
gradients/Merge_2:0
gradients/Merge_2:1
gradients/NextIteration_2:0
gradients/Switch_2:0
gradients/Switch_2:1
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter:0
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2:0
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc:0
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Enter:0
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPushV2:0
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc:0
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Enter:0
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPushV2:0
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc:0
rgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
xgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
rgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc:0
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/Enter:0
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPushV2:0
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/f_acc:0
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape:0
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Enter:0
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2:0
fgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc:0
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Enter:0
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPushV2:0
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc:0
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Enter:0
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPushV2:0
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc:0
Hgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape:0
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1:0
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0
\gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape:0
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1:0
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
fgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape:0
Mgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Shape:0
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Enter:0
Zgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPushV2:0
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc:0
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Enter:0
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc:0
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
Xgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape:0
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1:0
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Enter:0
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc:0
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0
Xgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape:0
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1:0
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
fgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
hgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Enter:0
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc:0
Jgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape:0
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1:0
gradients/f_count_3:0
gradients/f_count_4:0
gradients/f_count_5:0В
-blstm_1/bidirectional_rnn/bw/bw/TensorArray:0Qblstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0╚
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0_
'blstm_1/bidirectional_rnn/bw/bw/zeros:04blstm_1/bidirectional_rnn/bw/bw/while/Select/Enter:0░
Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0Vgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0g
1blstm_1/bidirectional_rnn/bw/bw/strided_slice_1:02blstm_1/bidirectional_rnn/bw/bw/while/Less/Enter:0и
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0Ь
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc:0Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Enter:0─
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0и
Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0Rgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0ш
rgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0rgradients/blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0k
-blstm_1/bidirectional_rnn/bw/bw/CheckSeqLen:0:blstm_1/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0v
4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/read:0>blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter:0д
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc:0Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Enter:0u
2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/read:0?blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter:0╚
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0м
Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc:0Tgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Enter:0╠
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0╚
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0a
)blstm_1/bidirectional_rnn/bw/bw/Minimum:04blstm_1/bidirectional_rnn/bw/bw/while/Less_1/Enter:0д
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc:0Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Enter:0д
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc:0Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Enter:0╚
bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0bgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0╠
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0д
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc:0Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Enter:0а
Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc:0Ngradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Enter:0Ь
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc:0Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter:0r
/blstm_1/bidirectional_rnn/bw/bw/TensorArray_1:0?blstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0└
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc:0^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/Enter:0Ь
Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/f_acc:0Lgradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/Enter:0─
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0`gradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0╠
dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0dgradients/blstm_1/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0б
\blstm_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Ablstm_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0д
Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc:0Pgradients/blstm_1/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Enter:0└
^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc:0^gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Enter:0─
`gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0`gradients/blstm_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0R-blstm_1/bidirectional_rnn/bw/bw/while/Enter:0R/blstm_1/bidirectional_rnn/bw/bw/while/Enter_1:0R/blstm_1/bidirectional_rnn/bw/bw/while/Enter_2:0R/blstm_1/bidirectional_rnn/bw/bw/while/Enter_3:0R/blstm_1/bidirectional_rnn/bw/bw/while/Enter_4:0Rgradients/f_count_4:0Z1blstm_1/bidirectional_rnn/bw/bw/strided_slice_1:0"Ь9
	variablesО9Л9
ы
/blstm_0/bidirectional_rnn/fw/lstm_cell/kernel:04blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Assign4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/read:02Jblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform:08
┌
-blstm_0/bidirectional_rnn/fw/lstm_cell/bias:02blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Assign2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/read:02?blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros:08
ы
/blstm_0/bidirectional_rnn/bw/lstm_cell/kernel:04blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Assign4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/read:02Jblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform:08
┌
-blstm_0/bidirectional_rnn/bw/lstm_cell/bias:02blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Assign2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/read:02?blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros:08
ы
/blstm_1/bidirectional_rnn/fw/lstm_cell/kernel:04blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Assign4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/read:02Jblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform:08
┌
-blstm_1/bidirectional_rnn/fw/lstm_cell/bias:02blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Assign2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/read:02?blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros:08
ы
/blstm_1/bidirectional_rnn/bw/lstm_cell/kernel:04blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Assign4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/read:02Jblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform:08
┌
-blstm_1/bidirectional_rnn/bw/lstm_cell/bias:02blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Assign2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/read:02?blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
Ї
4blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam:09blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Assign9blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam/read:02Fblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros:0
№
6blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1:0;blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Assign;blstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/read:02Hblstm_0/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros:0
ь
2blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam:07blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam/Assign7blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam/read:02Dblstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zeros:0
Ї
4blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1:09blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Assign9blstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/read:02Fblstm_0/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zeros:0
Ї
4blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam:09blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Assign9blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam/read:02Fblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros:0
№
6blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1:0;blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Assign;blstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/read:02Hblstm_0/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros:0
ь
2blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam:07blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam/Assign7blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam/read:02Dblstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zeros:0
Ї
4blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1:09blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Assign9blstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/read:02Fblstm_0/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zeros:0
Ї
4blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam:09blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Assign9blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/read:02Fblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros:0
№
6blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1:0;blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Assign;blstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/read:02Hblstm_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros:0
ь
2blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam:07blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam/Assign7blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam/read:02Dblstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zeros:0
Ї
4blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1:09blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Assign9blstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/read:02Fblstm_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zeros:0
Ї
4blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam:09blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Assign9blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/read:02Fblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros:0
№
6blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1:0;blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Assign;blstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/read:02Hblstm_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros:0
ь
2blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam:07blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam/Assign7blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam/read:02Dblstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zeros:0
Ї
4blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1:09blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Assign9blstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/read:02Fblstm_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zeros:0
p
dense/kernel/Adam:0dense/kernel/Adam/Assigndense/kernel/Adam/read:02%dense/kernel/Adam/Initializer/zeros:0
x
dense/kernel/Adam_1:0dense/kernel/Adam_1/Assigndense/kernel/Adam_1/read:02'dense/kernel/Adam_1/Initializer/zeros:0
h
dense/bias/Adam:0dense/bias/Adam/Assigndense/bias/Adam/read:02#dense/bias/Adam/Initializer/zeros:0
p
dense/bias/Adam_1:0dense/bias/Adam_1/Assigndense/bias/Adam_1/read:02%dense/bias/Adam_1/Initializer/zeros:0
x
dense_1/kernel/Adam:0dense_1/kernel/Adam/Assigndense_1/kernel/Adam/read:02'dense_1/kernel/Adam/Initializer/zeros:0
А
dense_1/kernel/Adam_1:0dense_1/kernel/Adam_1/Assigndense_1/kernel/Adam_1/read:02)dense_1/kernel/Adam_1/Initializer/zeros:0
p
dense_1/bias/Adam:0dense_1/bias/Adam/Assigndense_1/bias/Adam/read:02%dense_1/bias/Adam/Initializer/zeros:0
x
dense_1/bias/Adam_1:0dense_1/bias/Adam_1/Assigndense_1/bias/Adam_1/read:02'dense_1/bias/Adam_1/Initializer/zeros:0*╘
serving_default└
=
x8
inputs/Placeholder:0                  '
)
seq_len
inputs/Placeholder_2:08
logits.
dense/dense_1/Softmax:0         tensorflow/serving/predict