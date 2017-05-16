require 'torch'

metric = {}
function metric.precisionrecallvector(distance,labels,recallstep)
    assert(distance:nDimension() == 1) --distance: smaller is better
    assert(labels:nDimension() == 1)   --labels: +1 or -1
    assert(distance:isSameSizeAs(labels))
    --  precision-recall calculation
    local recallstep = recallstep or 0.1
    local sortvec, sortindex = distance:sort() -- ascending ordering distance
    local sort_label = labels:index(1,sortindex) --positive is based on the threshold
--    sortvec=sortvec:narrow(1,1,1000)
    sortindex=sortindex:narrow(1,1,1000)
    sort_label=sort_label:narrow(1,1,1000)
    local tp = sort_label:gt(0):float():cumsum()
    local fp = sort_label:lt(0):float():cumsum()
    local tp_fn = sort_label:eq(1):sum()
    local tp_fp = tp + fp
    local recall = torch.div(tp, tp_fn)
    local precision = tp:cdiv(tp_fp)

    --  average precision calculation
    local ap = 0
    local recallpoints = 0
    for i= 0,1,recallstep do
        recallpoints=recallpoints+1
    end

    local interval  = 1 / recallpoints

    for i = 0,1,recallstep do
        mask = recall:ge(i)
        local interpolated_p
        if mask:max() > 0 then
            interpolated_p = precision:maskedSelect(mask):max()
        else
            interpolated_p = 0 
        end
        ap = ap + interpolated_p * interval
    end
    return precision, recall, ap, sortindex
end

function  metric.precisionrecallmetrix(distance, labels, recallstep, test)
    assert(distance:nDimension() == 2)
    assert(labels:nDimension() == 2)
    assert(distance:isSameSizeAs(labels))
    local nretrieve = 1000             --map@1000
    local nsamples = distance:size(2)
    local nQuery = distance:size(1)
    local recall = torch.FloatTensor(nQuery,nretrieve)
    local precision = torch.FloatTensor(nQuery,nretrieve)
    local ap = torch.FloatTensor(nQuery)
    local sortindex = torch.FloatTensor(nQuery,nretrieve)
    for i = 1, nQuery do
        local _conf = distance:select(1,i)
        local _labels = labels:select(1,i)
        local _prec, _recall, _ap, _sortindex = metric.precisionrecallvector( _conf, _labels, recallstep)
        precision[i]:copy(_prec)
        recall[i]:copy(_recall)
        ap[i] =  _ap
        sortindex[i]:copy(_sortindex)
        print('Nth_query:',i,  'label:', test_label[i], 'ap:', _ap)
    end
    local map = ap:sum() / (ap:size(1))
    return precision, recall, ap, map, sortindex
end

function metric.precisionrecall(distance, labels, recallstep, test)
    local test = test
    if distance:nDimension() == 2 then
        return metric.precisionrecallmetrix(distance, labels, recallstep, test)
    elseif distance:nDimension() == 1 then
        return metric.precisionrecallvector(distance, labels, recallstep)
    else 
        error('vectors or matrices (nQuery samples) expected ! ')
    end
end

return metric